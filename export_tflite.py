"""
SekoKuva-Mobile-Net — Export to TFLite
=======================================
Copyright (c) 2026 BC Bertenex Oy

Converts the trained PyTorch model to TensorFlow Lite format for mobile deployment.
Two export modes:
  1. Feature extractor only (for on-device transfer learning in the app)
  2. Full classifier (for demo/testing purposes)

Pipeline: PyTorch → ONNX → TFLite (using onnx2tf, which is reliable and maintained)

Usage:
    pip install onnx onnx2tf tensorflow
    python export_tflite.py --checkpoint checkpoints/best.pt --mode features
    python export_tflite.py --checkpoint checkpoints/best.pt --mode classifier
"""

import argparse
import os
import sys
import subprocess

import torch

from sekokuva_mobile_net.model import SekoKuvaMobileNet


def check_dependencies():
    """Check and report which export dependencies are available."""
    available = {}
    
    try:
        import onnx
        available["onnx"] = True
    except ImportError:
        available["onnx"] = False
    
    try:
        import onnx2tf
        available["onnx2tf"] = True
    except ImportError:
        available["onnx2tf"] = False
    
    try:
        import tensorflow as tf
        available["tensorflow"] = True
    except ImportError:
        available["tensorflow"] = False
    
    return available


def export_to_onnx(model, input_size, output_path, mode="features"):
    """
    Step 1: PyTorch → ONNX
    
    ONNX (Open Neural Network Exchange) is a universal format that
    bridges PyTorch and TensorFlow. Think of it as a "PDF for neural networks" —
    any framework can read it.
    """
    import onnx
    
    model.eval()
    
    # Create a dummy input to trace the network
    dummy_input = torch.randn(1, 3, input_size, input_size)
    
    # Choose which output to export
    if mode == "features":
        # Wrap model to only output features
        class FeatureWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                return self.model.forward_features(x)
        
        export_model = FeatureWrapper(model)
    else:
        export_model = model
    
    # Export
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        opset_version=13,
        input_names=["input_image"],
        output_names=["output"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "output": {0: "batch_size"}
        }
    )
    print(f"  Exported ONNX model to: {output_path}")
    
    # Verify
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"  ONNX model verified ✓")


def convert_onnx_to_tflite(onnx_path, tflite_path, quantize=True):
    """
    Step 2: ONNX → TFLite (using onnx2tf)
    
    onnx2tf is a reliable, actively maintained converter that goes
    directly from ONNX to TFLite without the fragile onnx-tf package.
    
    We also apply INT8 quantization:
    - Converts 32-bit floating point weights → 8-bit integers
    - Reduces model size by ~4x
    - Often speeds up inference on mobile (hardware is optimized for int8)
    - Tiny accuracy loss (~0.5-1%)
    """
    import onnx2tf
    
    # Output directory for onnx2tf (it creates a saved_model inside)
    out_dir = onnx_path.replace(".onnx", "_tflite_out")
    
    # onnx2tf converts ONNX → SavedModel → TFLite in one step
    print("  Converting ONNX → TFLite via onnx2tf...")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=out_dir,
        non_verbose=True,
    )
    
    # onnx2tf places the float32 tflite in the output folder
    generated_tflite = os.path.join(out_dir, "model_float32.tflite")
    
    if quantize:
        # Apply dynamic range quantization using TFLite converter
        try:
            import tensorflow as tf
            
            print("  Applying dynamic range quantization...")
            converter = tf.lite.TFLiteConverter.from_saved_model(
                os.path.join(out_dir, "saved_model")
            )
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
            
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
        except Exception as e:
            print(f"  Quantization failed ({e}), using float32 version instead")
            if os.path.exists(generated_tflite):
                import shutil
                shutil.copy2(generated_tflite, tflite_path)
    else:
        # Just copy the float32 version
        import shutil
        if os.path.exists(generated_tflite):
            shutil.copy2(generated_tflite, tflite_path)
    
    if os.path.exists(tflite_path):
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
        print(f"  Exported TFLite model to: {tflite_path} ({size_mb:.1f} MB)")
    else:
        print(f"  ERROR: TFLite export failed. Check the onnx2tf output in {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export SekoKuva-Mobile-Net to TFLite")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained PyTorch checkpoint (.pt)")
    parser.add_argument("--output_dir", type=str, default="./exported",
                        help="Output directory for exported models")
    parser.add_argument("--mode", type=str, choices=["features", "classifier"], default="features",
                        help="Export mode: 'features' for transfer learning backbone, "
                             "'classifier' for full model with classification head")
    parser.add_argument("--quantize", action="store_true", default=True,
                        help="Apply INT8 quantization (smaller + faster)")
    parser.add_argument("--no-quantize", dest="quantize", action="store_false",
                        help="Export full float32 model")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check dependencies
    deps = check_dependencies()
    if not deps["onnx"]:
        print("Missing package: onnx")
        print("  pip install onnx")
        sys.exit(1)
    
    can_tflite = deps["onnx2tf"] and deps["tensorflow"]
    if not can_tflite:
        missing = [n for n in ["onnx2tf", "tensorflow"] if not deps[n]]
        print(f"Note: {', '.join(missing)} not installed — will export ONNX only.")
        print(f"  To also get TFLite: pip install onnx2tf tensorflow")
        print(f"  (Or convert on LUMI where TensorFlow is available)")
        print()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    num_classes = checkpoint["num_classes"]
    input_size = checkpoint.get("input_size", 224)
    
    # Recreate model and load weights
    model = SekoKuvaMobileNet(
        num_classes=num_classes,
        input_size=input_size
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"  Model loaded: {num_classes} classes, input size {input_size}")
    print(f"  Best validation accuracy was: {checkpoint.get('best_val_acc', 'N/A')}%")
    print(f"  Export mode: {args.mode}")
    print()
    
    # Step 1: PyTorch → ONNX (always works)
    onnx_path = os.path.join(args.output_dir, f"sekokuva_mobilenet_{args.mode}.onnx")
    export_to_onnx(model, input_size, onnx_path, mode=args.mode)
    print()
    
    # Step 2: ONNX → TFLite (only if tensorflow + onnx2tf are available)
    if can_tflite:
        suffix = "_quantized" if args.quantize else "_float32"
        tflite_path = os.path.join(args.output_dir, f"sekokuva_mobilenet_{args.mode}{suffix}.tflite")
        convert_onnx_to_tflite(onnx_path, tflite_path, quantize=args.quantize)
        print()
    
    print("=" * 60)
    print("  Export complete!")
    print(f"  ONNX:   {onnx_path}")
    if can_tflite:
        print(f"  TFLite: {tflite_path}")
    else:
        print(f"  TFLite: (convert on LUMI — run this script there with tensorflow installed)")
    print()
    print("  Next step: embed the .tflite file in the Android app")
    print("=" * 60)


if __name__ == "__main__":
    main()
