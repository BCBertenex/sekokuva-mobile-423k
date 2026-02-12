"""
SekoKuva-Mobile-Net — Test & Demo Script
==========================================
Copyright (c) 2026 BC Bertenex Oy

Test your trained model on any image! Works with:
  - PyTorch checkpoint (.pt)  → full classification with class names
  - ONNX model (.onnx)        → inference via onnxruntime (no PyTorch needed)

Usage:
    # Classify an image using the PyTorch checkpoint:
    python test_model.py --image photo.jpg

    # Classify an image using the ONNX model:
    python test_model.py --image photo.jpg --onnx exported/sekokuva_mobilenet_features.onnx

    # Use a webcam snapshot (requires opencv):
    python test_model.py --webcam

    # Test with a random noise image (just to verify the pipeline works):
    python test_model.py --random
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

# Image preprocessing — MUST match what training used
INPUT_SIZE = 224
# ImageNet normalization values (same as train.py)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess_image(image_path):
    """
    Load and preprocess an image for the model.

    Steps (same as validation in train.py):
      1. Resize so shortest side = 256
      2. Center crop to 224×224
      3. Convert to float [0, 1]
      4. Normalize with ImageNet mean/std
      5. Rearrange to CHW format (channels, height, width)
      6. Add batch dimension → [1, 3, 224, 224]
    """
    img = Image.open(image_path).convert("RGB")

    # Resize shortest side to 256, keeping aspect ratio
    w, h = img.size
    if w < h:
        new_w = 256
        new_h = int(h * 256 / w)
    else:
        new_h = 256
        new_w = int(w * 256 / h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    # Center crop to 224×224
    left = (new_w - INPUT_SIZE) // 2
    top = (new_h - INPUT_SIZE) // 2
    img = img.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))

    # To numpy, float32, [0,1]
    arr = np.array(img, dtype=np.float32) / 255.0

    # Normalize
    arr = (arr - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)

    # HWC → CHW
    arr = arr.transpose(2, 0, 1)

    # Add batch dimension → [1, 3, 224, 224]
    arr = np.expand_dims(arr, axis=0)

    return arr


def make_random_input():
    """Generate a random noise 'image' for pipeline testing."""
    arr = np.random.randn(1, 3, INPUT_SIZE, INPUT_SIZE).astype(np.float32)
    return arr


def load_class_names(path="checkpoints/class_names.json"):
    """Load the class names saved during training."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ─── PyTorch inference ───────────────────────────────────────────────────────

def test_pytorch(checkpoint_path, input_array, class_names=None):
    """Run inference using the PyTorch checkpoint (full classifier)."""
    import torch
    from sekokuva_mobile_net.model import SekoKuvaMobileNet

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    num_classes = ckpt.get("num_classes", 15)
    saved_input_size = ckpt.get("input_size", INPUT_SIZE)

    # Rebuild model
    model = SekoKuvaMobileNet(num_classes=num_classes, input_size=saved_input_size)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Run inference
    tensor = torch.from_numpy(input_array)
    with torch.no_grad():
        logits = model(tensor)               # [1, num_classes]
        probs = torch.softmax(logits, dim=1)  # convert to probabilities

    probs = probs.numpy()[0]  # [num_classes]

    # Also get features
    with torch.no_grad():
        features = model.forward_features(tensor)  # [1, 512]

    print("\n" + "=" * 60)
    print("  PyTorch Inference Results")
    print("=" * 60)
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Classes    : {num_classes}")
    best_acc = ckpt.get('best_val_acc', None)
    if best_acc is not None:
        # Stored as 0-100 percentage, not 0-1
        acc_str = f"{best_acc:.1f}%" if best_acc > 1 else f"{best_acc:.1%}"
        print(f"  Val acc    : {acc_str}")
    else:
        print(f"  Val acc    : unknown")
    print(f"  Feature vec: {features.shape[1]}-dimensional")
    print()

    # Top-5 predictions
    top5_idx = np.argsort(probs)[::-1][:5]
    print("  Top-5 Predictions:")
    print("  " + "-" * 40)
    for rank, idx in enumerate(top5_idx, 1):
        name = class_names[idx] if class_names else f"class_{idx}"
        bar = "█" * int(probs[idx] * 30)
        print(f"  {rank}. {name:<20s} {probs[idx]:6.1%}  {bar}")

    print()
    return probs, features.numpy()[0]


# ─── ONNX inference ──────────────────────────────────────────────────────────

def test_onnx(onnx_path, input_array, class_names=None):
    """Run inference using the ONNX model via onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n  onnxruntime not installed. Install it with:")
        print("    pip install onnxruntime")
        sys.exit(1)

    # Create session
    session = ort.InferenceSession(onnx_path)

    # Get input/output info
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_name = session.get_outputs()[0].name
    output_shape = session.get_outputs()[0].shape

    # Run inference
    result = session.run([output_name], {input_name: input_array})
    output = result[0][0]  # remove batch dimension

    print("\n" + "=" * 60)
    print("  ONNX Runtime Inference Results")
    print("=" * 60)
    print(f"  Model      : {onnx_path}")
    print(f"  Input      : {input_name} {input_shape}")
    print(f"  Output     : {output_name} {list(output_shape)}")
    print(f"  Output size: {len(output)} values")
    print()

    # If output is 512-dim → feature extractor mode
    if len(output) > 100:
        print("  This is a FEATURE EXTRACTOR model (not a classifier).")
        print(f"  Output: {len(output)}-dimensional feature vector")
        print()
        print("  Feature vector stats:")
        print(f"    Min   : {output.min():.4f}")
        print(f"    Max   : {output.max():.4f}")
        print(f"    Mean  : {output.mean():.4f}")
        print(f"    Std   : {output.std():.4f}")
        print(f"    Zeros : {np.sum(output == 0)} / {len(output)}  (ReLU killed)")
        print()
        print("  This is what the mobile app uses as input for on-device")
        print("  transfer learning — the 512 numbers that describe the image.")
    else:
        # Classification output — apply softmax
        exp_vals = np.exp(output - np.max(output))
        probs = exp_vals / exp_vals.sum()

        top5_idx = np.argsort(probs)[::-1][:5]
        print("  Top-5 Predictions:")
        print("  " + "-" * 40)
        for rank, idx in enumerate(top5_idx, 1):
            name = class_names[idx] if class_names else f"class_{idx}"
            bar = "█" * int(probs[idx] * 30)
            print(f"  {rank}. {name:<20s} {probs[idx]:6.1%}  {bar}")

    print()
    return output


# ─── Webcam ──────────────────────────────────────────────────────────────────

def capture_webcam(save_path="webcam_capture.jpg"):
    """Capture a frame from the webcam."""
    try:
        import cv2
    except ImportError:
        print("  opencv not installed. Install with: pip install opencv-python")
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  Could not open webcam!")
        sys.exit(1)

    print("  Webcam opened. Press SPACE to capture, ESC to cancel...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("SekoKuva - Press SPACE to capture", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            cv2.imwrite(save_path, frame)
            print(f"  Captured → {save_path}")
            break
        elif key == 27:  # ESC
            print("  Cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(0)

    cap.release()
    cv2.destroyAllWindows()
    return save_path


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test SekoKuva-Mobile-Net on an image"
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to an image file (jpg, png, etc.)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to PyTorch checkpoint (default: checkpoints/best.pt)"
    )
    parser.add_argument(
        "--onnx", type=str, default=None,
        help="Path to ONNX model (uses onnxruntime instead of PyTorch)"
    )
    parser.add_argument(
        "--webcam", action="store_true",
        help="Capture from webcam"
    )
    parser.add_argument(
        "--random", action="store_true",
        help="Use random noise (for pipeline testing)"
    )
    parser.add_argument(
        "--class-names", type=str, default="checkpoints/class_names.json",
        help="Path to class_names.json"
    )

    args = parser.parse_args()

    # ── Get input image ──
    if args.random:
        print("  Using random noise input (pipeline test)")
        input_array = make_random_input()
    elif args.webcam:
        path = capture_webcam()
        input_array = preprocess_image(path)
    elif args.image:
        if not os.path.exists(args.image):
            print(f"  File not found: {args.image}")
            sys.exit(1)
        print(f"  Image: {args.image}")
        input_array = preprocess_image(args.image)
    else:
        print("  No input specified. Use --image, --webcam, or --random")
        print("  Quick test:  python test_model.py --random")
        sys.exit(1)

    print(f"  Input tensor: {input_array.shape}  (batch, channels, height, width)")

    # ── Load class names ──
    class_names = load_class_names(args.class_names)
    if class_names:
        print(f"  Loaded {len(class_names)} class names")

    # ── Run inference ──
    if args.onnx:
        test_onnx(args.onnx, input_array, class_names)
    else:
        test_pytorch(args.checkpoint, input_array, class_names)

    # ── If both are available, also compare ──
    if args.onnx and os.path.exists(args.checkpoint):
        print("  TIP: Run without --onnx to also see PyTorch classification results")
    elif not args.onnx and os.path.exists("exported/sekokuva_mobilenet_features.onnx"):
        print("  TIP: Also test the ONNX model:")
        print("    python test_model.py --random --onnx exported/sekokuva_mobilenet_features.onnx")


if __name__ == "__main__":
    main()
