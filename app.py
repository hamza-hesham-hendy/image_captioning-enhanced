"""
app.py — Image Captioning Live Demo
Run: streamlit run app.py

Features:
  • Live camera capture  (point at real objects, get instant caption)
  • File upload  /  URL paste
  • Per-word 8-head attention heatmaps
  • Per-word GradCAM overlay
  • All-8-heads side-by-side view
  • BLEU / CIDEr / METEOR scoring against a reference
"""

import os, io, time
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
import streamlit as st
from PIL import Image

from models import (
    load_artifacts, get_device, load_checkpoint,
    beam_search, greedy_decode,
    per_word_attention, per_word_gradcam, per_word_all_heads,
    GradCAM, get_transform, denormalize,
    compute_bleu, CIDErScorer, meteor_score_single,
)

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Captioning Demo",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.word-btn {
    display:inline-block; margin:3px; padding:5px 12px;
    border-radius:20px; cursor:pointer; font-size:15px;
    border: 1.5px solid var(--primary-color);
    background: transparent; transition: all .15s;
}
.word-btn:hover { background: #e8f0fe; }
.word-btn.selected { background:#1a73e8; color:#fff; border-color:#1a73e8; }
.metric-row { display:flex; gap:16px; margin:8px 0; }
.metric-box { flex:1; text-align:center; padding:10px;
              border-radius:10px; background:#f8f9fa; }
.metric-val { font-size:22px; font-weight:600; }
.metric-lbl { font-size:12px; color:#666; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    model_choice = st.selectbox(
        "Model",
        ["GRU + 8-head Attention", "LSTM + 8-head Attention"],
    )
    beam_width = st.slider("Beam width", 1, 10, 5)
    alpha      = st.slider("Heatmap opacity", 0.1, 0.8, 0.45, 0.05)
    max_words  = st.slider("Words to visualise", 4, 16, 10)

    st.divider()
    viz_mode = st.radio(
        "Visualisation",
        ["Attention (fast)", "Attention + GradCAM", "All 8 heads"],
        index=0,
    )
    st.caption(
        "**Attention** — free from the forward pass.  \n"
        "**GradCAM** — one backward pass per word, slower.  \n"
        "**All heads** — show each of 8 heads separately."
    )

    st.divider()
    st.markdown("### Score against reference")
    ref_caption = st.text_area(
        "Reference caption (optional):",
        placeholder="a dog running on the grass",
        height=80,
    )

# ─────────────────────────────────────────────────────────────────
# Load model and vocab  (cached — only runs once)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_everything():
    device    = get_device()
    artifacts = load_artifacts("artifacts/eda_artifacts.pkl")
    word2idx  = artifacts["word2idx"]
    idx2word  = artifacts["idx2word"]
    vocab_size = artifacts["vocab_size"]

    models_loaded = {}
    paths = {
        "GRU + 8-head Attention":  ("gru_attn",  "saved_models/gru_attn_final.pth"),
        "LSTM + 8-head Attention": ("lstm_attn", "saved_models/lstm_attn_final.pth"),
    }
    for label, (mtype, path) in paths.items():
        if os.path.exists(path):
            m, _ = load_checkpoint(mtype, path, device, vocab_size,
                                   fine_tune_blocks=0)
            models_loaded[label] = m

    return device, word2idx, idx2word, models_loaded

device, word2idx, idx2word, models_loaded = load_everything()

if not models_loaded:
    st.error(
        "No trained model checkpoints found in `saved_models/`.  \n"
        "Train the models first using `02_modelling.ipynb`."
    )
    st.stop()

# Select active model
model = models_loaded.get(model_choice, list(models_loaded.values())[0])
tf_val = get_transform("val")

# ─────────────────────────────────────────────────────────────────
# Image input  —  3 sources
# ─────────────────────────────────────────────────────────────────
st.title("📸 Image Captioning — Live Demo")
st.caption(
    "EfficientNetV2-S  ·  8-head cross-attention  ·  GRU / LSTM  ·  "
    "GloVe 6B 300d  ·  COCO train2017"
)

tab_cam, tab_upload, tab_url = st.tabs(
    ["📷 Live Camera", "📁 Upload File", "🔗 Paste URL"]
)

raw_pil = None

with tab_cam:
    st.markdown(
        "**Point your camera at anything — the model will caption it.**  \n"
        "Press *Take Photo* to capture."
    )
    cam_img = st.camera_input("Take a photo")
    if cam_img is not None:
        raw_pil = Image.open(cam_img).convert("RGB")

with tab_upload:
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp"]
    )
    if uploaded is not None:
        raw_pil = Image.open(uploaded).convert("RGB")

with tab_url:
    url = st.text_input("Image URL", placeholder="https://…/image.jpg")
    if url.strip():
        try:
            resp = requests.get(url, timeout=8)
            raw_pil = Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            st.error(f"Could not load image from URL: {e}")

if raw_pil is None:
    st.info("↑  Choose an image source above to get started.")
    st.stop()

# ─────────────────────────────────────────────────────────────────
# Preprocess
# ─────────────────────────────────────────────────────────────────
# Resize shorter side → 384 → CenterCrop(384)  (no distortion)
img_t   = tf_val(raw_pil).to(device)
img_np  = np.array(raw_pil.resize(
    (384, int(raw_pil.height * 384 / min(raw_pil.size))),
    Image.BILINEAR
).crop((0, 0, 384, 384))) / 255.0   # for display

# ─────────────────────────────────────────────────────────────────
# Generate caption
# ─────────────────────────────────────────────────────────────────
with st.spinner("Generating caption…"):
    t0      = time.time()
    caption = beam_search(model, img_t, word2idx, idx2word,
                          beam_width=beam_width)
    elapsed = time.time() - t0

st.markdown(f"## 📝 &nbsp;{caption}")
st.caption(f"Generated in {elapsed:.2f}s  ·  beam width={beam_width}")

# ─────────────────────────────────────────────────────────────────
# Scoring (if reference given)
# ─────────────────────────────────────────────────────────────────
if ref_caption.strip():
    bleu   = compute_bleu([caption], [[ref_caption]])
    cider  = CIDErScorer().score([caption], [[ref_caption]])
    meteor = meteor_score_single(caption, [ref_caption]) or 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("BLEU-1", f"{bleu['BLEU-1']:.3f}")
    c2.metric("BLEU-4", f"{bleu['BLEU-4']:.3f}")
    c3.metric("CIDEr",  f"{cider:.3f}")
    c4.metric("METEOR", f"{meteor:.3f}")

st.divider()

# ─────────────────────────────────────────────────────────────────
# Helper: overlay heatmap onto image
# ─────────────────────────────────────────────────────────────────
def blend(img_np, heatmap, alpha=0.45):
    """Resize heatmap to 384×384 and blend with image."""
    h, w = img_np.shape[:2]
    hm   = cv2.resize(heatmap.astype(np.float32), (w, h),
                      interpolation=cv2.INTER_CUBIC)
    hm   = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    cmap = plt.cm.jet(hm)[:, :, :3]
    return np.clip((1 - alpha) * img_np + alpha * cmap, 0, 1)

def fig_to_pil(fig):
    """Convert matplotlib figure to PIL image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                dpi=110, facecolor="white")
    buf.seek(0)
    return Image.open(buf)

# ─────────────────────────────────────────────────────────────────
# Compute attention maps (always — they're free)
# ─────────────────────────────────────────────────────────────────
with st.spinner("Computing attention maps…"):
    attn_maps = per_word_attention(model, img_t, word2idx, idx2word)
    attn_maps = attn_maps[:max_words]

words = [w for w, _ in attn_maps]

# ─────────────────────────────────────────────────────────────────
# Main layout: image | word selector
# ─────────────────────────────────────────────────────────────────
col_img, col_ctrl = st.columns([1, 2], gap="large")

with col_img:
    st.image(raw_pil, caption="Input image", width='stretch')

with col_ctrl:
    st.markdown("#### Select a word to inspect")
    selected_word = st.radio(
        "Word", words, horizontal=True, label_visibility="collapsed"
    )
    sel_idx = words.index(selected_word) if selected_word in words else 0

    st.markdown(
        f"Showing spatial focus when the model generated **\"{selected_word}\"**"
    )

# ─────────────────────────────────────────────────────────────────
# Visualisation tabs
# ─────────────────────────────────────────────────────────────────
vt1, vt2, vt3, vt4 = st.tabs([
    "🎯 Attention", "🔥 GradCAM", "👁 All 8 Heads", "🖼 Caption Strip"
])

# ── Tab 1: Attention overlay ──────────────────────────────────────
with vt1:
    _, attn = attn_maps[sel_idx]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=11); axes[0].axis("off")

    raw_hm = cv2.resize(attn.astype(np.float32), (384, 384),
                        interpolation=cv2.INTER_CUBIC)
    axes[1].imshow(raw_hm, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"Attention map — '{selected_word}'", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(blend(img_np, attn, alpha))
    axes[2].set_title("Overlay", fontsize=11); axes[2].axis("off")

    plt.tight_layout()
    st.image(fig_to_pil(fig), width='stretch')
    plt.close(fig)

# ── Tab 2: GradCAM ───────────────────────────────────────────────
with vt2:
    compute_cam = st.button("Compute GradCAM for this word",
                             type="primary")
    st.caption("GradCAM runs one backward pass per word — takes ~2-5 seconds.")

    if compute_cam:
        with st.spinner(f"GradCAM for '{selected_word}'…"):
            gcam      = GradCAM(model)
            cam_maps  = per_word_gradcam(gcam, img_t, word2idx, idx2word)
            cam_maps  = cam_maps[:max_words]

        if sel_idx < len(cam_maps):
            _, cam = cam_maps[sel_idx]
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            axes[0].imshow(img_np)
            axes[0].set_title("Original", fontsize=11); axes[0].axis("off")
            raw_cam = cv2.resize(cam.astype(np.float32), (384, 384),
                                 interpolation=cv2.INTER_CUBIC)
            axes[1].imshow(raw_cam, cmap="jet", vmin=0, vmax=1)
            axes[1].set_title(f"GradCAM — '{selected_word}'", fontsize=11)
            axes[1].axis("off")
            axes[2].imshow(blend(img_np, cam, alpha=0.4))
            axes[2].set_title("Overlay", fontsize=11); axes[2].axis("off")
            plt.tight_layout()
            st.image(fig_to_pil(fig), width='stretch')
            plt.close(fig)
        else:
            st.warning("Word not found in GradCAM output.")

# ── Tab 3: All 8 heads ───────────────────────────────────────────
with vt3:
    st.markdown(
        "Each head attends to **different spatial aspects** of the image. "
        "Some heads track objects, others track spatial relations or textures."
    )
    with st.spinner("Fetching per-head weights…"):
        head_results = per_word_all_heads(
            model, img_t, word2idx, idx2word,
            target_word=selected_word
        )

    if head_results:
        word_found, head_maps = head_results[0]   # (8, 12, 12)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        for i, ax in enumerate(axes.flat):
            ax.imshow(blend(img_np, head_maps[i], alpha=0.5))
            ax.set_title(f"Head {i+1}", fontsize=11)
            ax.axis("off")
        plt.suptitle(
            f"All 8 attention heads — word: \"{word_found}\"",
            fontsize=13, y=1.01
        )
        plt.tight_layout()
        st.image(fig_to_pil(fig), width='stretch')
        plt.close(fig)
    else:
        st.info(
            f"Word '{selected_word}' not found at this decode step. "
            "Try another word from the caption."
        )

# ── Tab 4: Caption strip (all words at once) ─────────────────────
with vt4:
    st.markdown(
        "Every word in the generated caption shown left-to-right "
        "with its averaged attention heatmap."
    )
    n = len(attn_maps)
    if n == 0:
        st.info("No words to show.")
    else:
        fig, axes = plt.subplots(
            1, n + 1,
            figsize=(2.5 * (n + 1), 3.2),
            gridspec_kw={"wspace": 0.05}
        )

        # First panel: original image
        axes[0].imshow(img_np)
        axes[0].set_title("Original", fontsize=9)
        axes[0].axis("off")

        for j, (word, attn) in enumerate(attn_maps):
            axes[j + 1].imshow(blend(img_np, attn, alpha))
            fw = "bold" if word == selected_word else "normal"
            axes[j + 1].set_title(word, fontsize=10, fontweight=fw)
            axes[j + 1].axis("off")

        plt.suptitle(
            f"Caption: \"{' '.join(words)}\"",
            fontsize=10, y=1.02
        )
        st.image(fig_to_pil(fig), width='stretch')
        plt.close(fig)

# ─────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "**Architecture** — EfficientNetV2-S encoder (384×384, 144 patches) "
    "· 8-head cross-attention · GRU / LSTM decoder · GloVe 6B 300d embeddings  \n"
    "**Training** — MS-COCO train2017 (118k images) · label smoothing ε=0.1 "
    "· teacher forcing · cosine LR · Phase-1 cached + Phase-2 fine-tune"
)
