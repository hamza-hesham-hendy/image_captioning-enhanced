"""
models.py — Image Captioning
==============================
Encoder    : EfficientNetV2-S  (384×384 input, 144 spatial patches 12×12)
Decoder    : GRU / LSTM  +  8-head cross-attention
Embeddings : GloVe 6B 300d  →  Linear(300 → 512)
Dataset    : MS-COCO train2017  (118k images)
Training   : Phase-1 cached (frozen encoder) → Phase-2 live fine-tune (last 2 blocks)
Loss       : CrossEntropy  +  label smoothing ε=0.1  +  teacher forcing
"""

import os, pickle, time, math, collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from PIL import Image
from tqdm.auto import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────
# 1.  TRANSFORMS  — match EfficientNetV2-S pretraining exactly
# ─────────────────────────────────────────────────────────────────
IMG_SIZE    = 384      # EfficientNetV2-S was pretrained at 384×384
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
_BILINEAR = transforms.InterpolationMode.BILINEAR   # matches torchvision weights


def get_transform(split='train'):
    """
    Train  : RandomResizedCrop(384) preserves semantic content without squish.
    Val/Test: Resize shorter side → 384 (aspect-ratio safe), then CenterCrop.
              This exactly matches the EfficientNetV2-S evaluation protocol.
    """
    if split == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, interpolation=_BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    # val / test / inference
    return transforms.Compose([
        # single int → shorter side resized, AR preserved
        transforms.Resize(IMG_SIZE, interpolation=_BILINEAR),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(tensor):
    """Undo ImageNet normalisation → numpy HWC [0,1] for display."""
    m = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    s = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (tensor.cpu() * s + m).clamp(0, 1).permute(1, 2, 0)
    return img.numpy()


# ─────────────────────────────────────────────────────────────────
# 2.  ENCODER — EfficientNetV2-S
# ─────────────────────────────────────────────────────────────────
class EfficientNetV2Encoder(nn.Module):
    """EfficientNetV2-S spatial feature extractor.

    Input  : (B, 3, 384, 384)
    Output : (B, 144, feature_dim)   — 12×12 patches, projected to feature_dim

    EfficientNetV2-S has 8 feature blocks (indices 0–7):
      0 : stem conv
      1 : FusedMBConv stage (2 blocks)
      2 : FusedMBConv stage (4 blocks)
      3 : FusedMBConv stage (4 blocks)
      4 : MBConv stage (6 blocks)
      5 : MBConv stage (9 blocks)
      6 : MBConv stage (15 blocks)   ← richer semantic features
      7 : head conv → 1280ch         ← most abstract spatial map
    'Last 2 blocks' = features[-2] + features[-1]
    """
    N_PATCHES  = 144    # 12 × 12  at 384×384 input
    PATCH_SIZE = 12
    EFF_DIM    = 1280   # output channels of EfficientNetV2-S

    def __init__(self, feature_dim=512, fine_tune_blocks=0):
        super().__init__()
        eff = models.efficientnet_v2_s(
            weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        # Keep only the convolutional backbone (drop avgpool + classifier)
        self.features    = eff.features          # (B, 1280, 12, 12) at 384×384
        self.proj        = nn.Linear(self.EFF_DIM, feature_dim)
        self.feature_dim = feature_dim
        self.patch_size  = self.PATCH_SIZE

        # Freeze all convolutional weights by default
        for p in self.features.parameters():
            p.requires_grad = False

        # Optionally unfreeze last N blocks (for Phase 2)
        if fine_tune_blocks > 0:
            for block in list(self.features)[-fine_tune_blocks:]:
                for p in block.parameters():
                    p.requires_grad = True

    def forward(self, x):
        x = self.features(x)                       # (B, 1280, 12, 12)
        x = x.flatten(2).permute(0, 2, 1)          # (B, 144, 1280)
        return self.proj(x)                         # (B, 144, feature_dim)


def unfreeze_encoder_blocks(model, n_blocks=2):
    """Unfreeze last n_blocks of encoder for Phase-2 fine-tuning.
    Call this after loading a Phase-1 checkpoint.
    """
    for p in model.encoder.features.parameters():
        p.requires_grad = False
    for block in list(model.encoder.features)[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Phase-2: unfroze last {n_blocks} encoder blocks → "
          f"{trainable/1e6:.1f}M trainable params")


# ─────────────────────────────────────────────────────────────────
# 3.  GLOVE EMBEDDINGS
# ─────────────────────────────────────────────────────────────────
def load_glove(glove_path, word2idx, embed_dim=300):
    """Build embedding matrix from GloVe vectors.

    Words in vocabulary but not in GloVe are initialised with
    uniform noise in the same range as real GloVe vectors (~0.05).
    <pad> (index 0) is zeroed out.

    Returns : np.ndarray  (vocab_size, embed_dim)  float32
    """
    vocab_size = len(word2idx)
    # Random uniform in same range as GloVe (~[-0.05, 0.05])
    matrix = np.random.uniform(-0.05, 0.05,
                               (vocab_size, embed_dim)).astype(np.float32)
    matrix[0] = 0.0   # <pad> → zero vector

    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word  = parts[0]
            if word in word2idx:
                matrix[word2idx[word]] = np.array(
                    parts[1:], dtype=np.float32
                )
                found += 1

    pct = found / vocab_size * 100
    print(f"GloVe: {found:,}/{vocab_size:,} words found ({pct:.1f}%)")
    return matrix


# ─────────────────────────────────────────────────────────────────
# 4.  DECODER — GRU / LSTM with 8-head cross-attention
# ─────────────────────────────────────────────────────────────────
class MultiHeadAttnCaptioner(nn.Module):
    """GRU or LSTM decoder with multi-head cross-attention.

    At each timestep t  (teacher forcing during training):
      1. emb  = proj(GloVe_embed(w_{t-1}))           (B, units)
      2. h_t  = RNNCell(emb, h_{t-1})                (B, units)
      3. ctx  = MHA(query=h_t, key=feat, val=feat)   (B, units)
      4. out  = Linear(concat(h_t, ctx))              (B, vocab)

    Attention weights (B, 144) are available for free at inference —
    reshape to (12, 12) for per-word spatial heatmap.
    """
    def __init__(self, vocab_size, glove_matrix=None,
                 embed_dim=300, units=512, num_heads=8,
                 rnn_type='gru', dropout=0.4,
                 fine_tune_blocks=0):
        super().__init__()

        # Encoder
        self.encoder = EfficientNetV2Encoder(
            feature_dim=units,
            fine_tune_blocks=fine_tune_blocks
        )

        # GloVe embedding + projection  300 → 512
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if glove_matrix is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(glove_matrix)
            )
        self.emb_proj  = nn.Linear(embed_dim, units)
        self.emb_norm  = nn.LayerNorm(units)

        # RNN cell
        self.rnn_type = rnn_type
        Cell = nn.GRUCell if rnn_type == 'gru' else nn.LSTMCell
        self.rnn = Cell(units, units)

        # 8-head cross-attention
        assert units % num_heads == 0, "units must be divisible by num_heads"
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=units, num_heads=num_heads,
            dropout=0.1, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(units)

        # Output
        self.fc_out  = nn.Linear(units * 2, units)
        self.out_norm = nn.LayerNorm(units)
        self.fc_pred = nn.Linear(units, vocab_size)
        self.drop    = nn.Dropout(dropout)

        self.vocab_size = vocab_size
        self.units      = units
        self.num_heads  = num_heads
        self.embed_dim  = embed_dim

    # ── helpers ──────────────────────────────────────────────────
    def _init_hidden(self, feat):
        """Initialise hidden state from mean of image patches."""
        h = feat.mean(dim=1)                       # (B, units)
        if self.rnn_type == 'lstm':
            return h, torch.zeros_like(h)
        return h, None

    def _step(self, tok_emb, h, c, feat):
        """One decode step. Returns (logits, h_new, c_new, attn_weights)."""
        emb = self.emb_norm(self.emb_proj(tok_emb))   # (B, units)

        if self.rnn_type == 'gru':
            h = self.rnn(emb, h); c = None
        else:
            h, c = self.rnn(emb, (h, c))

        # Cross-attention
        q       = h.unsqueeze(1)                       # (B, 1, units)
        ctx, wt = self.cross_attn(q, feat, feat,
                                  need_weights=True,
                                  average_attn_weights=True)
        ctx = self.attn_norm(ctx.squeeze(1))           # (B, units)

        # Output
        combined = self.drop(torch.relu(
            self.out_norm(self.fc_out(torch.cat([h, ctx], dim=-1)))
        ))
        logits = self.fc_pred(combined)                # (B, vocab)
        # wt: (B, 1, 144) → (B, 144)
        return logits, h, c, wt.squeeze(1)

    # ── forward (teacher forcing) ─────────────────────────────────
    def forward(self, images, captions, features=None):
        """Teacher-forced forward pass.

        images   : (B, 3, 384, 384) — pass None when features is provided
        captions : (B, T) token ids  [<start>, w1, …, w_{T-1}]
        features : (B, 144, units)  pre-extracted (Phase-1 cached training)
        returns  : (B, T, vocab_size) logits
        """
        feat = features if features is not None else self.encoder(images)
        B, T = captions.shape
        emb  = self.embedding(captions)               # (B, T, embed_dim)

        h, c = self._init_hidden(feat)
        outputs = torch.zeros(B, T, self.vocab_size,
                              device=feat.device)

        for t in range(T):
            logits, h, c, _ = self._step(emb[:, t], h, c, feat)
            outputs[:, t]   = logits

        return outputs

    def attend(self, feat, h):
        """Single attention step for inference — returns (context, avg_weights).
        avg_weights : (B, 144)  averaged over 8 heads.
        """
        q = h.unsqueeze(1)
        ctx, wt = self.cross_attn(q, feat, feat,
                                  need_weights=True,
                                  average_attn_weights=True)
        return ctx.squeeze(1), wt.squeeze(1)   # (B, units), (B, 144)

    def attend_all_heads(self, feat, h):
        """Returns per-head weights (B, num_heads, 144) for visualization."""
        q = h.unsqueeze(1)
        _, wt = self.cross_attn(q, feat, feat,
                                need_weights=True,
                                average_attn_weights=False)
        return wt[:, :, 0, :]   # (B, num_heads, 144)


# ─────────────────────────────────────────────────────────────────
# 5.  DATASET
# ─────────────────────────────────────────────────────────────────
class CaptioningDataset(Dataset):
    """Live dataset — loads images from disk per batch.
    Used for Phase-2 fine-tuning and evaluation.
    DataFrame must have columns: image | input_seq | target_seq
    (image column holds the full absolute path)
    """
    def __init__(self, df, split='train'):
        self.df        = df.reset_index(drop=True)
        self.transform = get_transform(split)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['image']).convert('RGB')
        img = self.transform(img)
        inp = torch.tensor(row['input_seq'],  dtype=torch.long)
        tgt = torch.tensor(row['target_seq'], dtype=torch.long)
        return img, inp, tgt


class CachedFeatureDataset(Dataset):
    """Phase-1 dataset — returns pre-extracted features, no image I/O.
    features_arr : np.memmap or ndarray  (N_images, 144, feature_dim) float16
    path2idx     : dict  image_path → row index in features_arr
    """
    def __init__(self, df, features_arr, path2idx):
        self.df           = df.reset_index(drop=True)
        self.features_arr = features_arr
        self.path2idx     = path2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        fidx = self.path2idx[row['image']]
        feat = torch.from_numpy(
            self.features_arr[fidx].astype(np.float32)
        )
        inp  = torch.tensor(row['input_seq'],  dtype=torch.long)
        tgt  = torch.tensor(row['target_seq'], dtype=torch.long)
        return feat, inp, tgt


class SharedEncoderDataset(Dataset):
    """Phase-2 dataset that groups all captions per image together.

    Each __getitem__ returns (image_tensor, input_seqs, target_seqs) where
    input_seqs and target_seqs have shape (N_caps, T).

    The training loop encodes the image ONCE and expands features to all
    N_caps captions — giving the H100 a 5× larger effective batch with
    only 1/5th of the encoder calls.
    """
    def __init__(self, df, split='train'):
        self.transform = get_transform(split)
        # Group captions by image path
        grouped = df.groupby('image', sort=False)
        self.image_paths  = list(grouped.groups.keys())
        self.groups       = {p: grouped.get_group(p) for p in self.image_paths}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path  = self.image_paths[idx]
        grp   = self.groups[path]
        img   = Image.open(path).convert('RGB')
        img_t = self.transform(img)
        inp   = torch.tensor(grp['input_seq'].tolist(),  dtype=torch.long)
        tgt   = torch.tensor(grp['target_seq'].tolist(), dtype=torch.long)
        return img_t, inp, tgt   # (3,H,W), (N,T), (N,T)


def _shared_collate(batch):
    """Custom collate: stack images; pad caption lists to same N."""
    imgs, inps, tgts = zip(*batch)
    imgs = torch.stack(imgs)                       # (B, 3, H, W)
    max_n = max(x.size(0) for x in inps)
    T     = inps[0].size(1)
    inp_pad = torch.zeros(len(inps), max_n, T, dtype=torch.long)
    tgt_pad = torch.zeros(len(tgts), max_n, T, dtype=torch.long)
    for i, (inp, tgt) in enumerate(zip(inps, tgts)):
        n = inp.size(0)
        inp_pad[i, :n] = inp
        tgt_pad[i, :n] = tgt
    return imgs, inp_pad, tgt_pad


def build_shared_dataloaders(artifacts, batch_size=256, num_workers=8):
    """Phase-2 DataLoaders — shared encoder, one image processed N_caps times."""
    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=(num_workers > 0),
              prefetch_factor=4 if num_workers > 0 else None,
              collate_fn=_shared_collate)
    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = SharedEncoderDataset(artifacts[f'{split}_df'], split=split)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == 'train'), **kw
        )
    return loaders['train'], loaders['val'], loaders['test']


def load_artifacts(path='artifacts/eda_artifacts.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _loader_kw(num_workers=4):
    return dict(
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


def build_dataloaders(artifacts, batch_size=256, num_workers=4):
    """Live DataLoaders for Phase-2 fine-tuning and evaluation."""
    kw = _loader_kw(num_workers)
    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = CaptioningDataset(artifacts[f'{split}_df'], split=split)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == 'train'), **kw
        )
    return loaders['train'], loaders['val'], loaders['test']


def build_cached_dataloaders(artifacts, features_arr, path2idx,
                              batch_size=512, num_workers=4):
    """Cached DataLoaders for Phase-1 (no image I/O)."""
    kw = _loader_kw(num_workers)
    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = CachedFeatureDataset(
            artifacts[f'{split}_df'], features_arr, path2idx
        )
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == 'train'), **kw
        )
    return loaders['train'], loaders['val'], loaders['test']


# ─────────────────────────────────────────────────────────────────
# 6.  FEATURE CACHING
# ─────────────────────────────────────────────────────────────────
def extract_and_cache_features(encoder, all_df, cache_dir,
                                device, batch_size=256):
    """Run EfficientNetV2-S over every unique image once.

    Saves two files to cache_dir:
      features.npy  — (N, 144, feature_dim) float16  [mmap-friendly]
      index.pkl     — {image_path: row_index}

    Returns (features_arr, path2idx) ready for CachedFeatureDataset.
    """
    os.makedirs(cache_dir, exist_ok=True)
    feat_path  = os.path.join(cache_dir, 'features.npy')
    index_path = os.path.join(cache_dir, 'index.pkl')

    all_paths = all_df['image'].unique().tolist()
    N         = len(all_paths)
    D         = encoder.feature_dim   # 512
    P         = encoder.N_PATCHES     # 144

    if os.path.exists(feat_path) and os.path.exists(index_path):
        print(f"Cache exists — loading from {cache_dir}")
        arr      = np.load(feat_path, mmap_mode='r')
        path2idx = pickle.load(open(index_path, 'rb'))
        print(f"  {len(path2idx):,} images  "
              f"({arr.nbytes / 1e9:.1f} GB on disk)")
        return arr, path2idx

    print(f"Extracting features for {N:,} images → {cache_dir}")
    tf = get_transform('val')        # no augmentation for feature extraction
    encoder.eval()

    # Pre-allocate float16 array
    arr = np.zeros((N, P, D), dtype=np.float16)

    for i in tqdm(range(0, N, batch_size), desc='Extracting'):
        batch_paths = all_paths[i: i + batch_size]
        imgs, valid = [], []
        for j, p in enumerate(batch_paths):
            try:
                img = Image.open(p).convert('RGB')
                imgs.append(tf(img))
                valid.append(i + j)
            except Exception:
                pass
        if not imgs:
            continue
        with torch.no_grad():
            feats = encoder(torch.stack(imgs).to(device))   # (B, 144, 512)
        for k, idx in enumerate(valid):
            arr[idx] = feats[k].cpu().float().numpy().astype(np.float16)

    np.save(feat_path, arr)
    path2idx = {p: i for i, p in enumerate(all_paths)}
    pickle.dump(path2idx, open(index_path, 'wb'))

    size_gb = arr.nbytes / 1e9
    print(f"Saved: {feat_path}  ({size_gb:.1f} GB)")
    return np.load(feat_path, mmap_mode='r'), path2idx


# ─────────────────────────────────────────────────────────────────
# 7.  DECODING — greedy + beam search
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def _greedy_token_ids(model, img_t, word2idx, max_len=40):
    """Run greedy decode for one image. img_t: (1,3,H,W) on device."""
    model.eval()
    device = img_t.device
    feat   = model.encoder(img_t)              # (1, 144, units)
    h, c   = model._init_hidden(feat)
    tok    = word2idx['<start>']
    ids    = []
    for _ in range(max_len):
        emb = model.embedding(torch.tensor([[tok]], device=device))
        emb = emb[:, 0, :]                     # (1, embed_dim)
        logits, h, c, _ = model._step(emb, h, c, feat)
        tok = logits.argmax(-1).item()
        if tok == word2idx.get('<end>', -1):
            break
        ids.append(tok)
    return ids


@torch.no_grad()
def greedy_decode(model, image_tensor, word2idx, idx2word, max_len=40):
    img = image_tensor.unsqueeze(0) if image_tensor.dim() == 3 else image_tensor
    ids = _greedy_token_ids(model, img, word2idx, max_len)
    return ' '.join(idx2word.get(t, '') for t in ids).strip()


@torch.no_grad()
def beam_search(model, image_tensor, word2idx, idx2word,
                beam_width=5, max_len=40):
    """Beam search decoding for a single image tensor."""
    model.eval()
    device = image_tensor.device
    img    = (image_tensor.unsqueeze(0)
              if image_tensor.dim() == 3 else image_tensor)
    feat   = model.encoder(img)                # (1, 144, units)
    h0, c0 = model._init_hidden(feat)

    start  = word2idx['<start>']
    end    = word2idx['<end>']
    pad    = word2idx.get('<pad>', 0)

    # Each beam: (log_prob, token_ids, h, c)
    beams  = [(0.0, [start], h0, c0)]
    done   = []

    for _ in range(max_len):
        candidates = []
        for score, seq, h, c in beams:
            last_tok = torch.tensor([[seq[-1]]], device=device)
            emb = model.embedding(last_tok)[:, 0, :]
            logits, h_new, c_new, _ = model._step(emb, h, c, feat)
            log_p = torch.log_softmax(logits, dim=-1).squeeze(0)
            topk  = torch.topk(log_p, beam_width)
            for k in range(beam_width):
                tok = topk.indices[k].item()
                s   = score + topk.values[k].item()
                if tok == end:
                    done.append((s / (len(seq)), seq + [tok]))
                else:
                    candidates.append((s, seq + [tok], h_new, c_new))
        if not candidates:
            break
        candidates.sort(key=lambda x: -x[0])
        beams = candidates[:beam_width]

    if not done:
        done = [(b[0] / len(b[1]), b[1]) for b in beams]
    best = max(done, key=lambda x: x[0])[1]
    words = [idx2word.get(t, '') for t in best
             if t not in (start, end, pad)]
    return ' '.join(w for w in words if w).strip()


# ─────────────────────────────────────────────────────────────────
# 8.  GRAD-CAM
# ─────────────────────────────────────────────────────────────────
class GradCAM:
    """Gradient-weighted Class Activation Maps for EfficientNetV2-S.

    Hooks on the last convolutional block (features[-1], output 1280×12×12).
    Standard GradCAM: channel-weighted spatial sum → (12, 12) map.
    """
    def __init__(self, model):
        self.model       = model
        self.activations = None
        self.gradients   = None
        # Hook on last EfficientNetV2-S feature block
        target = model.encoder.features[-1]
        target.register_forward_hook(self._fwd)
        target.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.activations = o.detach()          # (B, 1280, 12, 12)

    def _bwd(self, m, gi, go):
        self.gradients = go[0].detach()        # (B, 1280, 12, 12)

    def _compute_cam(self):
        acts  = self.activations[0]            # (1280, 12, 12)
        grads = self.gradients[0]              # (1280, 12, 12)
        w     = grads.mean(dim=[1, 2])         # (1280,)
        cam   = (w[:, None, None] * acts).sum(0)  # (12, 12)
        cam   = torch.relu(cam)
        cam   = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy()

    def generate(self, image_tensor, caption_input):
        """GradCAM for the full caption (sum of all word scores)."""
        self.model.eval()
        img = (image_tensor.unsqueeze(0)
               if image_tensor.dim() == 3 else image_tensor)
        cap = (caption_input.unsqueeze(0)
               if caption_input.dim() == 1 else caption_input)
        img = img.detach().requires_grad_(True)
        out   = self.model(img, cap)
        score = out[0].gather(
            1, cap[0, :out.size(1)].unsqueeze(1)
        ).sum()
        self.model.zero_grad()
        score.backward()
        return self._compute_cam()

    def generate_for_word(self, image_tensor, cap_input,
                          word_position, word_idx):
        """GradCAM for one specific predicted word."""
        self.model.eval()
        img = (image_tensor.unsqueeze(0)
               if image_tensor.dim() == 3 else image_tensor)
        cap = (cap_input.unsqueeze(0)
               if cap_input.dim() == 1 else cap_input)
        img = img.detach().requires_grad_(True)
        out   = self.model(img, cap)
        score = out[0, word_position, word_idx]
        self.model.zero_grad()
        score.backward()
        return self._compute_cam()


# ─────────────────────────────────────────────────────────────────
# 9.  PER-WORD VISUALISATION
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def per_word_attention(model, image_tensor, word2idx, idx2word,
                       max_len=40):
    """Return averaged 8-head attention weights per predicted word.

    Returns : list of (word_str, attn_np_12x12)
    No backward pass — weights are free from the attention forward call.
    """
    model.eval()
    device = image_tensor.device
    img    = (image_tensor.unsqueeze(0)
              if image_tensor.dim() == 3 else image_tensor)
    ps     = model.encoder.patch_size   # 12

    feat  = model.encoder(img)
    h, c  = model._init_hidden(feat)
    tok   = word2idx['<start>']
    result = []

    for _ in range(max_len):
        emb = model.embedding(
            torch.tensor([[tok]], device=device)
        )[:, 0, :]
        logits, h, c, weights = model._step(emb, h, c, feat)
        tok = logits.argmax(-1).item()
        if tok == word2idx.get('<end>', -1):
            break
        word = idx2word.get(tok, '<unk>')
        attn = weights[0].cpu().numpy()         # (144,)
        attn = attn.reshape(ps, ps)             # (12, 12)
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
        result.append((word, attn))

    return result


@torch.no_grad()
def per_word_all_heads(model, image_tensor, word2idx, idx2word,
                       target_word=None, max_len=40):
    """Return all 8 attention heads for each predicted word (or just target_word).

    Returns : list of (word_str, head_maps_8x12x12)
    """
    model.eval()
    device = image_tensor.device
    img    = (image_tensor.unsqueeze(0)
              if image_tensor.dim() == 3 else image_tensor)
    ps     = model.encoder.patch_size   # 12

    feat  = model.encoder(img)
    h, c  = model._init_hidden(feat)
    tok   = word2idx['<start>']
    result = []

    for _ in range(max_len):
        emb = model.embedding(
            torch.tensor([[tok]], device=device)
        )[:, 0, :]
        logits, h, c, _ = model._step(emb, h, c, feat)

        # Per-head weights
        heads = model.attend_all_heads(feat, h)  # (1, 8, 144)
        head_maps = heads[0].cpu().numpy()        # (8, 144)
        head_maps = head_maps.reshape(
            model.num_heads, ps, ps
        )                                         # (8, 12, 12)

        tok  = logits.argmax(-1).item()
        if tok == word2idx.get('<end>', -1):
            break
        word = idx2word.get(tok, '<unk>')

        if target_word is None or word == target_word:
            result.append((word, head_maps))
        if target_word and word == target_word:
            break

    return result


def per_word_gradcam(gradcam, image_tensor, word2idx, idx2word,
                     max_len=40):
    """GradCAM heatmap for each predicted word.

    Returns : list of (word_str, heatmap_12x12)
    One forward+backward per word — slower than attention but gradient-based.
    """
    model  = gradcam.model
    device = image_tensor.device
    img    = (image_tensor.unsqueeze(0)
              if image_tensor.dim() == 3 else image_tensor)

    # Step 1 — greedy decode (no grad)
    # pred_ids = _greedy_token_ids(model, img, word2idx, max_len)
    
    # Step 1 — beam-search final caption only
    caption = beam_search(model, image_tensor, word2idx, idx2word, beam_width=5, max_len=max_len)

    # Convert final caption words back to token ids
    pred_ids = [word2idx[w] for w in caption.split() if w in word2idx]

    # Step 2 — GradCAM per word
    result = []
    for pos, tok_id in enumerate(pred_ids):
        word = idx2word.get(tok_id, '<unk>')
        if word in ('<end>', '<pad>'):
            break
        input_ids = [word2idx['<start>']] + pred_ids[:pos]
        cap = torch.tensor(input_ids, dtype=torch.long, device=device)
        hm  = gradcam.generate_for_word(image_tensor, cap, pos, tok_id)
        result.append((word, hm))

    return result


# ─────────────────────────────────────────────────────────────────
# 10. EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────
def compute_bleu(hypotheses, references, max_n=4):
    """BLEU-1…4 for a corpus. references: list of list of str."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    sf   = SmoothingFunction().method1
    refs = [[r.split() for r in rlist] for rlist in references]
    hyps = [h.split() for h in hypotheses]
    out  = {}
    for n in range(1, max_n + 1):
        w = tuple([1 / n] * n)
        out[f'BLEU-{n}'] = corpus_bleu(refs, hyps, weights=w,
                                        smoothing_function=sf)
    return out


class CIDErScorer:
    """Lightweight TF-IDF CIDEr-D (no COCO API needed)."""
    def __init__(self, n=4):
        self.n = n

    def _ngrams(self, tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def score(self, hypotheses, references):
        """
        hypotheses : list[str]
        references : list[list[str]]
        Returns    : float  mean CIDEr score
        """
        hyp_toks = [h.split() for h in hypotheses]
        ref_toks = [[r.split() for r in rs] for rs in references]
        N = len(hypotheses)
        all_scores = []

        for n in range(1, self.n + 1):
            # Document frequency
            df = collections.Counter()
            for refs in ref_toks:
                seen = set()
                for r in refs:
                    for ng in set(self._ngrams(r, n)):
                        if ng not in seen:
                            df[ng] += 1
                            seen.add(ng)

            step = []
            for i, (hyp, refs) in enumerate(zip(hyp_toks, ref_toks)):
                # Hypothesis TF-IDF
                tf_h = collections.Counter(self._ngrams(hyp, n))
                tot  = sum(tf_h.values()) or 1
                hvec = {ng: (c/tot) * math.log(N/(df.get(ng,0)+1e-10))
                        for ng, c in tf_h.items()}

                # Reference TF-IDF (average over refs)
                rvec: dict = collections.defaultdict(float)
                for r in refs:
                    tf_r = collections.Counter(self._ngrams(r, n))
                    rt   = sum(tf_r.values()) or 1
                    for ng, c in tf_r.items():
                        rvec[ng] += (c/rt)*math.log(N/(df.get(ng,0)+1e-10))
                for ng in rvec:
                    rvec[ng] /= len(refs)

                nh = math.sqrt(sum(v**2 for v in hvec.values()) + 1e-10)
                nr = math.sqrt(sum(v**2 for v in rvec.values()) + 1e-10)
                dot = sum(hvec.get(ng,0)*rv for ng,rv in rvec.items())
                cos = dot / (nh * nr + 1e-10)

                ref_lens = [len(r) for r in refs]
                best_rl  = min(ref_lens, key=lambda l: abs(l - len(hyp)))
                lp = math.exp(min(0, 1 - best_rl / (len(hyp) + 1e-10)))
                step.append(lp * cos)
            all_scores.append(step)

        combined = [
            10 * sum(all_scores[n][i] for n in range(self.n)) / self.n
            for i in range(N)
        ]
        return float(np.mean(combined))


def meteor_score_single(hypothesis, references):
    """METEOR via nltk (optional — returns None if nltk not installed)."""
    try:
        from nltk.translate.meteor_score import meteor_score as _ms
        import nltk
        try:
            nltk.data.find('wordnet')
        except Exception:
            nltk.download('wordnet', quiet=True)
        return _ms([r.split() for r in references], hypothesis.split())
    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────
# 11. TRAINING UTILITIES
# ─────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        dev = torch.device('cuda')
        torch.backends.cudnn.benchmark     = True
        torch.set_float32_matmul_precision('high')
        props = torch.cuda.get_device_properties(dev)
        print(f"GPU : {props.name}")
        print(f"VRAM: {props.total_memory/1e9:.0f} GB")
        return dev
    print("CPU mode")
    return torch.device('cpu')


def _make_optimizer(model, lr, fine_tune_lr_scale=0.1, wd=1e-4):
    """Two param groups: encoder fine-tuned params get 10× lower LR."""
    enc_ids = {id(p) for p in model.encoder.parameters()
               if p.requires_grad}
    enc_params   = [p for p in model.parameters()
                    if p.requires_grad and id(p) in enc_ids]
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in enc_ids]
    groups = [{'params': other_params, 'lr': lr}]
    if enc_params:
        groups.append({'params': enc_params,
                       'lr': lr * fine_tune_lr_scale})
    return optim.AdamW(groups, lr=lr, weight_decay=wd)


def _train_step(model, batch, criterion, optimizer, scaler,
                device, use_amp, grad_accum, step, cached):
    x, inp, tgt = batch
    x   = x.to(device,  non_blocking=True)
    inp = inp.to(device, non_blocking=True)
    tgt = tgt.to(device, non_blocking=True)

    with torch.autocast('cuda', torch.bfloat16,
                         enabled=use_amp and device.type == 'cuda'):
        if cached:
            out = model(None, inp, features=x)
        else:
            out = model(x, inp)
        loss = criterion(
            out.reshape(-1, out.size(-1)), tgt.reshape(-1)
        ) / grad_accum

    scaler.scale(loss).backward()
    if (step + 1) % grad_accum == 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    return loss.item() * grad_accum


def train_one_epoch(model, loader, criterion, optimizer, scaler,
                    device, use_amp=True, grad_accum=1, cached=False):
    model.train()
    total = 0.0
    optimizer.zero_grad()
    for step, batch in enumerate(tqdm(loader, desc='Train', leave=False)):
        total += _train_step(model, batch, criterion, optimizer,
                             scaler, device, use_amp, grad_accum,
                             step, cached)
    return total / len(loader)


@torch.no_grad()
def evaluate_loss(model, loader, criterion, device,
                  use_amp=True, cached=False):
    model.eval()
    total = 0.0
    for batch in tqdm(loader, desc='Val', leave=False):
        x, inp, tgt = batch
        x   = x.to(device,  non_blocking=True)
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        with torch.autocast('cuda', torch.bfloat16,
                             enabled=use_amp and device.type == 'cuda'):
            out  = (model(None, inp, features=x)
                    if cached else model(x, inp))
            loss = criterion(
                out.reshape(-1, out.size(-1)), tgt.reshape(-1)
            )
        total += loss.item()
    return total / len(loader)


def train_model(model, train_loader, val_loader, device,
                epochs=15, lr=5e-4, use_amp=True,
                label_smoothing=0.1, grad_accum=1,
                cached=False, use_wandb=False, run_name='model'):
    """Full training loop with cosine LR and label smoothing.

    cached=True  → Phase-1 (features pre-extracted, encoder frozen)
    cached=False → Phase-2 (live images, encoder partially unfrozen)
    """
    criterion = nn.CrossEntropyLoss(
        ignore_index=0, label_smoothing=label_smoothing
    )
    optimizer = _make_optimizer(model, lr)
    scaler    = torch.amp.GradScaler(
        enabled=use_amp and device.type == 'cuda'
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    history  = {'train_loss': [], 'val_loss': []}
    best_val = float('inf')
    best_sd  = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tl = train_one_epoch(model, train_loader, criterion, optimizer,
                             scaler, device, use_amp, grad_accum, cached)
        vl = evaluate_loss(model, val_loader, criterion, device,
                           use_amp, cached)
        scheduler.step()
        history['train_loss'].append(tl)
        history['val_loss'].append(vl)
        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:>2}/{epochs} | "
              f"train {tl:.4f} | val {vl:.4f} | lr {lr_now:.2e}")

        if use_wandb:
            try:
                import wandb
                wandb.log({'train_loss': tl, 'val_loss': vl,
                           'lr': lr_now, 'epoch': epoch})
            except Exception:
                pass

        if vl < best_val:
            best_val = vl
            best_sd  = {k: v.cpu().clone()
                        for k, v in model.state_dict().items()}
            print(f"   ✓ best val: {best_val:.4f}")

    if best_sd:
        model.load_state_dict(best_sd)
    history['elapsed_s']     = time.time() - t0
    history['best_val_loss'] = best_val
    return history


# ─────────────────────────────────────────────────────────────────
# 12. MODEL FACTORY + CHECKPOINTS
# ─────────────────────────────────────────────────────────────────
def build_model(model_type, vocab_size, glove_matrix=None,
                embed_dim=300, units=512, num_heads=8,
                dropout=0.4, fine_tune_blocks=0):
    """
    model_type : 'gru_attn' | 'lstm_attn'
    Pass glove_matrix (np.ndarray) to initialise embeddings from GloVe.
    """
    kw = dict(
        vocab_size       = vocab_size,
        glove_matrix     = glove_matrix,
        embed_dim        = embed_dim,
        units            = units,
        num_heads        = num_heads,
        dropout          = dropout,
        fine_tune_blocks = fine_tune_blocks,
    )
    if model_type == 'gru_attn':
        return MultiHeadAttnCaptioner(**kw, rnn_type='gru')
    elif model_type == 'lstm_attn':
        return MultiHeadAttnCaptioner(**kw, rnn_type='lstm')
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. "
                         "Choose 'gru_attn' or 'lstm_attn'.")


def save_checkpoint(model, history, config, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    raw = getattr(model, '_orig_mod', model)   # unwrap torch.compile
    torch.save({
        'model_state_dict': raw.state_dict(),
        'history': history,
        'config':  config,
    }, path)
    print(f"Saved → {path}")


def load_checkpoint(model_type, path, device, vocab_size,
                    embed_dim=300, units=512, num_heads=8,
                    dropout=0.4, fine_tune_blocks=0):
    """Load model for inference (fine_tune_blocks=0 keeps encoder frozen)."""
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    cfg   = ckpt.get('config', {})
    model = build_model(
        model_type,
        vocab_size       = vocab_size,
        glove_matrix     = None,         # loaded from state_dict
        embed_dim        = cfg.get('embed_dim',  embed_dim),
        units            = cfg.get('units',      units),
        num_heads        = cfg.get('num_heads',  num_heads),
        dropout          = cfg.get('dropout',    dropout),
        fine_tune_blocks = fine_tune_blocks,
    )
    model.load_state_dict(ckpt['model_state_dict'])
    return model.to(device).eval(), ckpt.get('history', {})
