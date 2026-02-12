# SekoKuva MobileNet 423K

A lightweight, 100% open-source image classification model designed for on-device transfer learning.

🇫🇮 Designed and trained in Eurajoki, Finland by [BC Bertenex Oy](https://bertenex.com)

|  |  |
|---|---|
| **Parameters** | 423,020 (0.42M) |
| **Feature Dimension** | 512 |
| **Input** | 224 × 224 × 3 (RGB) |
| **Top-1 Accuracy** | 67.9% (65 classes) |
| **Model Size** | 1.2 MB (PyTorch) · 0.4 MB (int8 quantized) |
| **Training Data** | OpenImages V7, bbox-verified (CC BY 4.0) |
| **License** | Apache 2.0 |

## Why This Model?

Most pre-trained mobile models carry weights derived from ImageNet under ambiguous license terms. **SekoKuva MobileNet 423K has a 100% clean license chain** — architecture, training code, training data, and weights are all openly and permissively licensed. You can use it in commercial products with full legal clarity.

## Pre-trained Weights

**Download from HuggingFace:** [BCBertenex/sekokuva-mobilenet-423k](https://huggingface.co/BCBertenex/sekokuva-mobilenet-423k)

The HuggingFace repo contains:
- PyTorch checkpoints (`pytorch/best.pt`, `pytorch/swa.pt`)
- ONNX models (classifier + feature extractor)
- Full model card with architecture details, benchmarks, and usage examples

## Quick Start

```bash
# Clone this repo
git clone https://github.com/BCBertenex/sekokuva-mobile-net-423k.git
cd sekokuva-mobile-net-423k

# Install dependencies
pip install torch torchvision onnx onnxruntime numpy pillow tqdm

# Download pre-trained weights from HuggingFace
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('BCBertenex/sekokuva-mobilenet-423k', local_dir='./pretrained')
"
```

### Classify an Image

```python
import torch
from PIL import Image
from torchvision import transforms
from sekokuva_mobile_net.model import SekoKuvaMobileNet

# Load model
checkpoint = torch.load("pretrained/pytorch/swa.pt", map_location="cpu")
model = SekoKuvaMobileNet(num_classes=65)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Preprocess
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img = Image.open("photo.jpg").convert("RGB")
input_tensor = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    probs = torch.softmax(model(input_tensor), dim=1)
    top5 = torch.topk(probs, 5)

import json
with open("pretrained/class_names.json") as f:
    class_names = json.load(f)

for prob, idx in zip(top5.values[0], top5.indices[0]):
    print(f"  {class_names[idx]:<20s} {prob:.1%}")
```

### Extract Features (for Transfer Learning)

```python
with torch.no_grad():
    features = model.forward_features(input_tensor)  # [1, 512]
```

This 512-dim feature vector is the primary output — use it to build custom classifiers with as few as 5–10 images per class, entirely on-device.

## Train From Scratch

Reproduce the model using only open data:

```bash
# 1. Download training data (OpenImages V7, bbox-verified, CC BY 4.0)
python prepare_data_clean.py --download --preset diverse --max-per-class 1000

# 2. Train (all enhancements enabled by default)
python train.py --data_dir ./data/openimages_clean --epochs 200 --batch_size 128 --num_workers 4

# 3. Export to ONNX
python export_tflite.py --checkpoint checkpoints/best.pt --mode features
python export_tflite.py --checkpoint checkpoints/best.pt --mode classifier
```

### Training Features

All enabled by default, disable with `--no_xxx` flags:

- **Mixed Precision (AMP)** — ~2× speedup on GPUs with Tensor Cores
- **CutMix + MixUp** — advanced augmentation (+2–5% accuracy)
- **Class-Balanced Sampling** — ensures underrepresented classes get equal training time
- **Progressive Resolution** — 112px → 160px → 224px across training
- **Stochastic Weight Averaging (SWA)** — flatter minimum, better generalization
- **Gradient Accumulation** — configurable effective batch size

## Architecture

Depthwise separable convolutions (MobileNetV1-style) arranged in 5 stages:

```
Input: 224×224×3 (RGB)
  → Stage 1: Conv2d 3→32, stride 2          → 112×112×32
  → Stage 2: DepthwiseSeparable 32→64        →  56×56×64
  → Stage 3: DepthwiseSeparable 64→128→128   →  28×28×128
  → Stage 4: DepthwiseSeparable 128→256→256  →  14×14×256
  → Stage 5: DepthwiseSeparable 256→512      →   7×7×512
  → Global Average Pooling                   → 512-dim features
  → Optional: Linear classifier              → num_classes
```

ReLU6 activation, BatchNorm, dropout 0.2. See [model.py](sekokuva_mobile_net/model.py) for the full annotated implementation.

## Repository Structure

```
sekokuva-mobile-net-423k/
├── sekokuva_mobile_net/
│   ├── __init__.py
│   └── model.py              ← Full architecture (395 lines, heavily commented)
├── train.py                   ← Training pipeline with 6 enhancement techniques
├── export_tflite.py           ← PyTorch → ONNX (→ TFLite) export
├── prepare_data_clean.py      ← OpenImages V7 bbox-verified data downloader
├── test_model.py              ← Test/demo script (PyTorch + ONNX inference)
├── requirements.txt
├── LICENSE                    ← Apache 2.0
└── README.md
```

## Roadmap

| Model | Parameters | Status |
|---|---|---|
| **SekoKuva MobileNet 423K** | **423K** | **✅ Released** |
| SekoKuva MobileNet 5M | ~5M | 🔨 In development |

## License

Apache 2.0 — see [LICENSE](LICENSE) for full text.

Training data: [OpenImages V7](https://storage.googleapis.com/openimages/web/index.html) (CC BY 4.0).

## Citation

```bibtex
@misc{sekokuva2026mobilenet423k,
  title     = {SekoKuva MobileNet 423K: A Lightweight Open-Source Feature Extractor for On-Device Transfer Learning},
  author    = {{BC Bertenex Oy}},
  year      = {2026},
  url       = {https://huggingface.co/BCBertenex/sekokuva-mobilenet-423k},
  note      = {Apache 2.0 License. Trained on OpenImages V7 (CC BY 4.0).}
}
```

---

*Built by [BC Bertenex Oy](https://bertenex.com), Eurajoki, Finland*
