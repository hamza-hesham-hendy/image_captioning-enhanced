# Image Captioning v2

**EfficientNetV2-S + 8-head Cross-Attention + GRU / LSTM**

| | |
|---|---|
| **Encoder** | EfficientNetV2-S · 384×384 input · 144 patches (12×12) |
| **Preprocessing** | Resize shorter side → CenterCrop (no distortion) · bilinear |
| **Decoder** | GRU-Attn and LSTM-Attn · 8-head cross-attention |
| **Embeddings** | GloVe 6B 300d → Linear(300→512) · min freq=3 · trainable |
| **Loss** | CrossEntropy + label smoothing ε=0.1 · teacher forcing |
| **Dataset** | MS-COCO train2017 · 118k images · 90/5/5 split by image |
| **Hardware** | Lightning AI H100 80GB |

## Training strategy

```
Phase 1 (15 epochs, batch=2048)
  └─ EfficientNetV2-S fully frozen
  └─ Features extracted once → cached to disk (~17 GB float16)
  └─ Decoder trains on cached features

Phase 2 (10 epochs, batch=512)
  └─ Load Phase-1 checkpoint
  └─ Unfreeze last 2 EfficientNetV2-S blocks
  └─ Train end-to-end live · encoder LR = 5e-5 · decoder LR = 5e-4
```

## Estimated credit cost (H100 @ 3.01 credits/h)

| Phase | Time | Credits |
|---|---|---|
| EDA + GloVe + download | ~45 min | 2.3 |
| Feature extraction | ~16 min | 0.8 |
| Phase 1 × 2 models | ~62 min | 3.1 |
| Phase 2 × 2 models | ~409 min | 20.5 |
| Evaluation + viz | ~30 min | 1.5 |
| **Total** | **~8.1h** | **~28.2 credits** |

## Run order

```bash
pip install -r requirements.txt

# 1. Download COCO + GloVe, build vocab, save artifacts
jupyter notebook 01_eda.ipynb

# 2. Extract features → Phase-1 training → Phase-2 fine-tune
jupyter notebook 02_modelling.ipynb

# 3. BLEU / CIDEr / METEOR evaluation
jupyter notebook 03_model_analysis.ipynb

# 4. Per-word attention + GradCAM visualisation
jupyter notebook 04_visualization.ipynb

# 5. Live demo (camera / upload / URL)
streamlit run app.py
```

## Environment

- Create and activate the workspace virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell (Windows)
# or for cmd.exe: \.\.venv\\Scripts\\activate.bat
# or for bash/mac: source .venv/bin/activate
pip install -r requirements.txt
```

- A smaller runtime dependency set is available in `requirements-core.txt` if you want
  a lighter install for the app.

- GPU users: install a PyTorch wheel matching your CUDA version instead of the generic
  `torch` wheel. Example for CUDA 12.1 (replace with your CUDA):

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
```

- OpenCV: this repo uses `opencv-python-headless` (server-friendly). If you need
  direct camera access on your machine, install `opencv-python` instead.


## Visualisation

- **Per-word attention** — 8-head cross-attention weights reshaped to 12×12,
  overlaid on image for each generated word. Free from the forward pass.
- **Per-word GradCAM** — gradient of the logit for each word w.r.t.
  the last EfficientNetV2-S conv block's activation map.
- **All-heads view** — each of 8 attention heads shown separately,
  revealing different spatial focuses (objects, relations, textures).

## Demo

https://imagecaptioning-enhanced-hd3epvage6odrfej7bp6yn.streamlit.app/
