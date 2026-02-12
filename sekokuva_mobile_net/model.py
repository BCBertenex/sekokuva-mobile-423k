"""
SekoKuva-Mobile-Net — Architecture Definition
==============================================
Copyright (c) 2026 BC Bertenex Oy
License: Apache 2.0

A custom lightweight convolutional neural network designed for on-device
image classification and transfer learning on mobile phones.

Designed for the Uganda AI Education Pilot by BC Bertenex Oy (Finland).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# BUILDING BLOCK: Depthwise Separable Convolution
# =============================================================================
#
# This is the core trick that makes mobile-friendly networks possible.
#
# A normal convolution does TWO things at once:
#   1. Looks at spatial patterns (what's near what in the image)
#   2. Combines information across channels (mixing color/feature info)
#
# A depthwise separable convolution SPLITS these into two cheaper steps:
#   Step A — "Depthwise": Look at spatial patterns in EACH channel separately
#   Step B — "Pointwise": Mix channels together using tiny 1×1 convolutions
#
# Why is this cheaper? Math:
#   Normal conv (3×3, 128 in → 128 out): 128 × 128 × 3 × 3 = 147,456 operations
#   Depthwise separable:
#     Depthwise:  128 × 1 × 3 × 3  =  1,152 operations
#     Pointwise:  128 × 128 × 1 × 1 = 16,384 operations
#     Total:                           17,536 operations  ← ~8× cheaper!
#
# Same quality of features, fraction of the compute. This is why phones can run it.

class DepthwiseSeparableConv(nn.Module):
    """
    One building block of SekoKuva-Mobile-Net.
    
    Think of it as one "processing step" that:
    1. Looks for spatial patterns (edges, textures, shapes) — depthwise conv
    2. Mixes those patterns together into richer features — pointwise conv
    3. Normalizes the output — batch normalization
    4. Applies a non-linearity — ReLU6 activation
    
    Parameters
    ----------
    in_channels : int
        How many feature maps come IN (from the previous layer)
    out_channels : int
        How many feature maps go OUT (to the next layer)
    stride : int
        1 = keep the same spatial size
        2 = cut spatial size in half (112×112 → 56×56)
            This is how the network progressively shrinks the image
            while building up richer features.
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        # ---- Step A: Depthwise Convolution ----
        # Each input channel gets its OWN 3×3 filter.
        # "groups=in_channels" means: don't mix channels, process each one alone.
        # This is what makes it "depthwise" — it only looks at spatial patterns.
        self.depthwise = nn.Conv2d(
            in_channels,            # number of input feature maps
            in_channels,            # output same number (one filter per channel)
            kernel_size=3,          # 3×3 filter — looks at a 3×3 pixel neighborhood
            stride=stride,          # 1 = same size, 2 = halve the dimensions
            padding=1,              # pad edges so we don't lose border pixels
            groups=in_channels,     # THIS is the key — one filter per channel
            bias=False              # no bias needed, batch norm handles it
        )
        
        # Batch Normalization after depthwise
        # This stabilizes training by normalizing the values flowing through.
        # Without it, values can explode or vanish, making training fail.
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # ---- Step B: Pointwise Convolution ----
        # A 1×1 convolution that mixes information ACROSS channels.
        # This is where the network learns "if I see edge A AND texture B
        # together, that probably means feature C."
        self.pointwise = nn.Conv2d(
            in_channels,            # all channels come in
            out_channels,           # we can change the number of channels here
            kernel_size=1,          # 1×1 — only mixes channels, no spatial look
            stride=1,               # never changes spatial size
            padding=0,              # no padding needed for 1×1
            bias=False
        )
        
        # Batch Normalization after pointwise
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process one batch of images/features through this block.
        
        x shape: [batch_size, in_channels, height, width]
        output:  [batch_size, out_channels, height/stride, width/stride]
        """
        # Step A: Depthwise — spatial patterns per channel
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu6(x, inplace=True)   # ReLU6 caps values at 6 — prevents
                                         # extreme values, good for mobile quantization
        
        # Step B: Pointwise — mix channels
        x = self.pointwise(x)
        x = self.bn2(x)
        x = F.relu6(x, inplace=True)
        
        return x


# =============================================================================
# THE FULL NETWORK: SekoKuva-Mobile-Net
# =============================================================================
#
# Below is the complete architecture. It's a sequence of layers that
# progressively transform a raw photo into a compact "fingerprint"
# (feature vector) that describes what's in the image.
#
# The journey of a photo through the network:
#
#   224×224×3  (raw photo: 224 pixels tall, 224 wide, 3 color channels RGB)
#       ↓
#   112×112×32  (found edges and basic gradients)
#       ↓
#   56×56×64   (found textures: smooth, rough, striped, dotted)
#       ↓
#   28×28×128  (found parts: leaf edges, round shapes, color patches)
#       ↓
#   14×14×256  (found objects: whole leaves, fruits, faces)
#       ↓
#   7×7×512    (found high-level concepts: "plant", "healthy", "person")
#       ↓
#   512        (one vector of 512 numbers — the image's "fingerprint")
#
# This fingerprint is what the kids' transfer learning layer connects to.
# They train a small layer on top that says:
#   "fingerprint like THIS → healthy leaf"
#   "fingerprint like THAT → diseased leaf"

class SekoKuvaMobileNet(nn.Module):
    """
    SekoKuva-Mobile-Net — A lightweight feature extractor for mobile devices.
    
    Designed by BC Bertenex Oy for the Uganda AI Education Pilot.
    
    Parameters
    ----------
    num_classes : int
        Number of output categories. 
        - During pre-training: set to the number of OpenImages categories (e.g., 300)
        - For the app (feature extraction mode): set to 0, which removes the 
          classification head and outputs raw 512-dim features.
    input_size : int
        Expected input image size (square). Default 224.
    dropout_rate : float
        Dropout probability before the final classifier. Helps prevent overfitting.
    """
    
    def __init__(self, num_classes: int = 300, input_size: int = 224, dropout_rate: float = 0.2):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # =====================================================================
        # STAGE 1: First Contact
        # =====================================================================
        # The very first layer is a regular (non-depthwise) convolution.
        # Why? Because the input only has 3 channels (RGB). Depthwise on 3
        # channels would only create 3 filters — way too few to find anything
        # useful. So we use a normal conv to expand from 3 → 32 channels.
        #
        # stride=2 halves the image immediately: 224×224 → 112×112
        # This early downsampling saves enormous compute in all following layers.
        
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                3,                  # RGB input
                32,                 # expand to 32 feature maps
                kernel_size=3,      # 3×3 filter
                stride=2,           # halve spatial dimensions immediately
                padding=1,          # preserve edges
                bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        # Output: 112 × 112 × 32
        # The network now "sees" the image through 32 different lenses,
        # each looking for different low-level patterns.
        
        # =====================================================================
        # STAGE 2: Low-Level Features (edges, gradients, simple textures)
        # =====================================================================
        # One block that doubles the channels: 32 → 64
        # stride=2 shrinks spatial: 112×112 → 56×56
        
        self.stage2 = DepthwiseSeparableConv(32, 64, stride=2)
        # Output: 56 × 56 × 64
        
        # =====================================================================
        # STAGE 3: Mid-Level Features (textures, patterns, small shapes)
        # =====================================================================
        # Two blocks at 128 channels.
        # First block: stride=2 shrinks 56×56 → 28×28
        # Second block: stride=1 stays at 28×28 (adds depth, not shrinkage)
        #
        # Why two blocks? One block can learn one "step" of abstraction.
        # Two blocks can learn more complex combinations. It's like saying:
        # "First find circles and lines, then find circles MADE OF lines."
        
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
        )
        # Output: 28 × 28 × 128
        
        # =====================================================================
        # STAGE 4: High-Level Features (object parts, structures)
        # =====================================================================
        # Two blocks at 256 channels.
        # First block: stride=2 shrinks 28×28 → 14×14
        # Second block: stride=1 refines at 14×14
        #
        # At this stage, the network is recognizing things like:
        # "leaf with veins", "round fruit shape", "spotted pattern on surface"
        
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
        )
        # Output: 14 × 14 × 256
        
        # =====================================================================
        # STAGE 5: Semantic Features (whole objects, scenes, concepts)
        # =====================================================================
        # Final spatial processing. One block, 512 channels.
        # stride=2 shrinks 14×14 → 7×7
        #
        # These 512 feature maps encode high-level understanding:
        # "this is a plant", "this looks healthy", "there's a person here"
        
        self.stage5 = DepthwiseSeparableConv(256, 512, stride=2)
        # Output: 7 × 7 × 512
        
        # =====================================================================
        # POOLING: Collapse spatial dimensions
        # =====================================================================
        # Global Average Pooling takes each of the 512 feature maps (7×7 grids)
        # and averages all 49 values into a single number.
        #
        # Result: 512 numbers. One number per feature map.
        # This is the "fingerprint" — a 512-dimensional description of the image.
        #
        # Why average and not something else?
        #   - It's translation invariant: the leaf can be anywhere in the photo
        #   - It's fixed size: works regardless of input resolution
        #   - No learnable parameters: nothing to overfit
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # Output: 1 × 1 × 512 → squeezed to just 512
        
        # =====================================================================
        # CLASSIFIER HEAD (only used during pre-training)
        # =====================================================================
        # During pre-training on OpenImages, we need to classify into categories
        # so the network can learn. This is a simple linear layer.
        #
        # In the app, we REMOVE THIS and replace it with the kids' own
        # transfer learning layer.
        
        self.dropout = nn.Dropout(p=dropout_rate)
        
        if num_classes > 0:
            self.classifier = nn.Linear(512, num_classes)
        else:
            self.classifier = None  # Feature extraction mode
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features only — no classification.
        This is what the mobile app uses.
        
        Input:  [batch, 3, 224, 224] — batch of RGB images
        Output: [batch, 512] — batch of feature vectors
        """
        x = self.first_conv(x)     # [B, 32, 112, 112]
        x = self.stage2(x)         # [B, 64, 56, 56]
        x = self.stage3(x)         # [B, 128, 28, 28]
        x = self.stage4(x)         # [B, 256, 14, 14]
        x = self.stage5(x)         # [B, 512, 7, 7]
        x = self.global_pool(x)    # [B, 512, 1, 1]
        x = x.flatten(1)           # [B, 512]
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass — features + classification.
        Used during pre-training.
        
        Input:  [batch, 3, 224, 224]
        Output: [batch, num_classes] — log-probabilities per class
        """
        x = self.forward_features(x)   # [B, 512]
        x = self.dropout(x)
        
        if self.classifier is not None:
            x = self.classifier(x)      # [B, num_classes]
        
        return x


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sekokuva_mobilenet(num_classes: int = 300, **kwargs) -> SekoKuvaMobileNet:
    """Create a SekoKuva-Mobile-Net for pre-training."""
    return SekoKuvaMobileNet(num_classes=num_classes, **kwargs)


def sekokuva_mobilenet_features(**kwargs) -> SekoKuvaMobileNet:
    """Create a SekoKuva-Mobile-Net in feature extraction mode (for the app)."""
    return SekoKuvaMobileNet(num_classes=0, **kwargs)


# =============================================================================
# MODEL SUMMARY — run this file directly to see the architecture
# =============================================================================

if __name__ == "__main__":
    # Create the model
    model = sekokuva_mobilenet(num_classes=300)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=" * 60)
    print("  SekoKuva-Mobile-Net")
    print("  Designed by BC Bertenex Oy")
    print("=" * 60)
    print(f"  Total parameters:     {total_params:>10,}")
    print(f"  Trainable parameters: {trainable_params:>10,}")
    print(f"  Estimated model size: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")
    print(f"  Estimated TFLite:     {total_params * 1 / 1024 / 1024:.1f} MB (int8 quantized)")
    print()
    
    # Test with a dummy image
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Full forward (classification mode)
    output = model(dummy_input)
    print(f"  Input shape:          {list(dummy_input.shape)}")
    print(f"  Output shape (class): {list(output.shape)}")
    
    # Feature extraction mode
    features = model.forward_features(dummy_input)
    print(f"  Output shape (feat):  {list(features.shape)}")
    print()
    
    # Per-stage breakdown
    print("  Architecture breakdown:")
    print("  ─────────────────────────────────────────────")
    x = dummy_input
    stages = [
        ("First Conv (3→32)", model.first_conv),
        ("Stage 2 (32→64)", model.stage2),
        ("Stage 3 (64→128→128)", model.stage3),
        ("Stage 4 (128→256→256)", model.stage4),
        ("Stage 5 (256→512)", model.stage5),
    ]
    for name, stage in stages:
        x = stage(x)
        params = sum(p.numel() for p in stage.parameters())
        print(f"  {name:<25} → {str(list(x.shape)):>25}  ({params:>8,} params)")
    
    print("  ─────────────────────────────────────────────")
    print()
    print("  ✓ Ready for training on LUMI")
    print("  ✓ Ready for TFLite export")
    print("  ✓ Ready for on-device transfer learning")
