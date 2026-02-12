"""
SekoKuva-Mobile-Net — Clean OpenImages V7 Data Downloader
===========================================================
Copyright (c) 2026 BC Bertenex Oy

IMPROVED VERSION: Downloads only images with BOUNDING BOX annotations
from OpenImages V7. This ensures every image actually contains a visible
instance of the object — no mislabels, no brand confusion, no noise.

Why bounding boxes?
  - If someone drew a box around an apple in the image, there IS an apple.
  - Image-level labels are noisy (machine-generated, ambiguous).
  - Bounding box annotations are human-verified.

We crop the bounding box region + some padding, giving us clean, 
object-centric training images.

Usage:
    # See available categories (600 boxable classes)
    python prepare_data_clean.py --list-classes

    # Download a small test set
    python prepare_data_clean.py --download --preset small --max-per-class 100

    # Download full training set for LUMI
    python prepare_data_clean.py --download --preset diverse --max-per-class 500

Requirements:
    pip install fiftyone Pillow tqdm
"""

import argparse
import os
import random
from pathlib import Path

# ============================================================================
# CURATED CATEGORIES — Using specific MIDs to avoid ambiguity
# ============================================================================
# Each entry: (human_name, OpenImages_MID)
# The MID is the unique machine identifier that avoids confusion like
# "Apple" (fruit) vs "Apple" (brand).
#
# You can find MIDs at: https://storage.googleapis.com/openimages/web/visualizer/index.html

PRESET_DIVERSE = [
    # ---- Fruits (very relevant for Uganda) ----
    ("Apple_fruit", "Apple"),
    ("Banana", "Banana"),
    ("Orange_fruit", "Orange"),
    ("Mango", "Mango"),
    ("Grape", "Grape"),
    ("Strawberry", "Strawberry"),
    ("Pineapple", "Pineapple"),
    ("Coconut", "Coconut"),
    ("Watermelon", "Watermelon"),
    ("Lemon", "Lemon"),
    ("Peach", "Peach"),
    ("Pear", "Pear"),
    
    # ---- Vegetables ----
    ("Tomato", "Tomato"),
    ("Potato", "Potato"),
    ("Carrot", "Carrot"),
    ("Broccoli", "Broccoli"),
    ("Cabbage", "Cabbage"),
    
    # ---- Plants & Nature ----
    ("Flower", "Flower"),
    ("Rose", "Rose"),
    ("Houseplant", "Houseplant"),
    ("Tree", "Tree"),
    ("Palm_tree", "Palm tree"),
    ("Mushroom", "Mushroom"),
    
    # ---- Animals ----
    ("Cat", "Cat"),
    ("Dog", "Dog"),
    ("Bird", "Bird"),
    ("Chicken", "Chicken"),
    ("Cattle", "Cattle"),
    ("Goat", "Goat"),
    ("Sheep", "Sheep"),
    ("Fish", "Fish"),
    ("Horse", "Horse"),
    ("Elephant", "Elephant"),
    ("Butterfly", "Butterfly"),
    ("Frog", "Frog"),
    ("Tortoise", "Tortoise"),
    
    # ---- People ----
    ("Person", "Person"),
    ("Boy", "Boy"),
    ("Girl", "Girl"),
    ("Man", "Man"),
    ("Woman", "Woman"),
    
    # ---- Common Objects ----
    ("Book", "Book"),
    ("Pen", "Pen"),
    ("Mobile_phone", "Mobile phone"),
    ("Laptop", "Laptop"),
    ("Bicycle", "Bicycle"),
    ("Car", "Car"),
    ("Bus", "Bus"),
    ("Bottle", "Bottle"),
    ("Cup", "Cup"),
    ("Bowl", "Bowl"),
    ("Table", "Table"),
    ("Chair", "Chair"),
    ("Backpack", "Backpack"),
    ("Umbrella", "Umbrella"),
    ("Hat", "Hat"),
    ("Shoe", "Shoe"),
    ("Ball", "Ball"),
    ("Knife", "Knife"),
    ("Plate", "Plate"),
    
    # ---- Food ----
    ("Bread", "Bread"),
    ("Egg", "Egg"),
    ("Cake", "Cake"),
    ("Cookie", "Cookie"),
    
    # ---- Vehicles ----
    ("Motorcycle", "Motorcycle"),
    ("Truck", "Truck"),
    ("Boat", "Boat"),
    ("Airplane", "Airplane"),
    
    # ---- Nature scenes ----
    ("River", "River"),
    ("Waterfall", "Waterfall"),
]

PRESET_SMALL = [
    ("Apple_fruit", "Apple"),
    ("Banana", "Banana"),
    ("Orange_fruit", "Orange"),
    ("Tomato", "Tomato"),
    ("Flower", "Flower"),
    ("Cat", "Cat"),
    ("Dog", "Dog"),
    ("Bird", "Bird"),
    ("Car", "Car"),
    ("Bottle", "Bottle"),
    ("Book", "Book"),
    ("Chair", "Chair"),
    ("Person", "Person"),
    ("Ball", "Ball"),
    ("Bicycle", "Bicycle"),
]

PRESETS = {
    "diverse": PRESET_DIVERSE,
    "small": PRESET_SMALL,
}


def download_with_fiftyone_bbox(categories, max_per_class, output_dir, val_split=0.15):
    """
    Download images using FiftyOne, but ONLY images that have bounding box
    annotations. Then crop the bounding box region (with padding) so each
    output image shows exactly one object.
    
    This eliminates:
    - Images where the object is tiny in the background
    - Mislabeled images (machine-generated labels)
    - Brand confusion (Apple fruit vs Apple phone)
    """
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz
    except ImportError:
        print("FiftyOne not installed. Run: pip install fiftyone")
        return
    
    from PIL import Image
    from tqdm import tqdm
    
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  SekoKuva-Mobile-Net — Clean Data Download")
    print(f"  Using BOUNDING BOX annotations only (human-verified)")
    print(f"{'='*60}")
    print(f"  Categories:     {len(categories)}")
    print(f"  Max per class:  {max_per_class}")
    print(f"  Output:         {output_dir}")
    print(f"  Val split:      {val_split*100:.0f}%")
    print()
    
    for folder_name, oi_class_name in categories:
        print(f"━━━ {folder_name} (OpenImages: '{oi_class_name}') ━━━")
        
        # Create class folders
        train_class_dir = os.path.join(train_dir, folder_name.lower())
        val_class_dir = os.path.join(val_dir, folder_name.lower())
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Skip if already have enough
        existing = len(os.listdir(train_class_dir)) + len(os.listdir(val_class_dir))
        if existing >= max_per_class * 0.8:
            print(f"  Already have {existing} images, skipping.\n")
            continue
        
        dataset_name = f"sekokuva_{folder_name.lower()}"
        
        try:
            # Delete existing dataset if it exists (from previous failed run)
            if fo.dataset_exists(dataset_name):
                fo.delete_dataset(dataset_name)
            
            # Download with DETECTIONS (bounding boxes), not classifications
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split="train",
                label_types=["detections"],       # ← KEY: bounding boxes only
                classes=[oi_class_name],
                max_samples=max_per_class * 2,    # download extra, we'll filter
                dataset_name=dataset_name,
            )
            
            # Process each image
            count = 0
            image_paths = []
            
            for sample in tqdm(dataset, desc=f"  Processing"):
                if sample.filepath is None or not os.path.exists(sample.filepath):
                    continue
                
                # Get bounding boxes for our target class
                if sample.ground_truth is None:
                    continue
                
                detections = [
                    det for det in sample.ground_truth.detections
                    if det.label == oi_class_name
                ]
                
                if not detections:
                    continue
                
                try:
                    img = Image.open(sample.filepath).convert("RGB")
                    img_w, img_h = img.size
                except Exception:
                    continue
                
                for det_idx, det in enumerate(detections):
                    if count >= max_per_class:
                        break
                    
                    # Get bounding box (FiftyOne uses [x, y, w, h] normalized)
                    x, y, w, h = det.bounding_box
                    
                    # Convert to pixel coordinates with PADDING
                    # We add 20% padding around the box so the object has context
                    pad_w = w * 0.2
                    pad_h = h * 0.2
                    
                    x1 = max(0, x - pad_w) * img_w
                    y1 = max(0, y - pad_h) * img_h
                    x2 = min(1, x + w + pad_w) * img_w
                    y2 = min(1, y + h + pad_h) * img_h
                    
                    # Skip tiny boxes (probably annotation errors)
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area < 50 * 50:  # smaller than 50x50 pixels
                        continue
                    
                    # Crop the bounding box region
                    cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))
                    
                    # Skip if crop is too small
                    if cropped.size[0] < 32 or cropped.size[1] < 32:
                        continue
                    
                    filename = f"{folder_name.lower()}_{count:05d}.jpg"
                    image_paths.append((cropped, filename))
                    count += 1
                
                if count >= max_per_class:
                    break
            
            # Shuffle and split into train/val
            random.shuffle(image_paths)
            val_count = int(len(image_paths) * val_split)
            
            for i, (img, filename) in enumerate(image_paths):
                if i < val_count:
                    dst = os.path.join(val_class_dir, filename)
                else:
                    dst = os.path.join(train_class_dir, filename)
                img.save(dst, "JPEG", quality=92)
            
            # Clean up
            if fo.dataset_exists(dataset_name):
                fo.delete_dataset(dataset_name)
            
            train_count = len(os.listdir(train_class_dir))
            val_count_actual = len(os.listdir(val_class_dir))
            print(f"  ✓ {folder_name}: {train_count} train + {val_count_actual} val (cropped to bounding box)\n")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print(f"    '{oi_class_name}' might not have box annotations. Skipping.\n")
            if fo.dataset_exists(dataset_name):
                fo.delete_dataset(dataset_name)
    
    # Summary
    print("=" * 60)
    print("  Download complete!")
    print("=" * 60)
    
    total_train = 0
    total_val = 0
    num_classes = 0
    
    for d in sorted(os.listdir(train_dir)):
        td = os.path.join(train_dir, d)
        vd = os.path.join(val_dir, d)
        if os.path.isdir(td):
            tc = len(os.listdir(td))
            vc = len(os.listdir(vd)) if os.path.isdir(vd) else 0
            if tc + vc > 0:
                num_classes += 1
                total_train += tc
                total_val += vc
                print(f"  {d:<25} {tc:>5} train  {vc:>5} val")
    
    print(f"  {'─'*45}")
    print(f"  {'TOTAL':<25} {total_train:>5} train  {total_val:>5} val")
    print(f"  Classes: {num_classes}")
    print()
    print(f"  Ready to train:")
    print(f"    python train.py --data_dir {output_dir} --epochs 100")


def list_boxable_classes():
    """Show the 600 classes that have bounding box annotations (the clean ones)."""
    from urllib.request import urlretrieve
    import csv
    
    # Download the boxable class list
    url = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
    local_path = "class-descriptions-boxable.csv"
    
    if not os.path.exists(local_path):
        print("Downloading boxable class list...")
        urlretrieve(url, local_path)
    
    classes = []
    with open(local_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                classes.append((row[0], row[1]))
    
    print(f"\nOpenImages V7 — {len(classes)} classes with bounding box annotations")
    print("These are CLEAN — every image has a human-drawn box around the object.\n")
    
    for mid, name in sorted(classes, key=lambda x: x[1]):
        print(f"  {name:<40} ({mid})")
    
    print(f"\nTotal: {len(classes)} boxable classes")
    print(f"\nUse these names with --categories:")
    print(f'  python prepare_data_clean.py --download --categories "Apple,Banana,Dog"')


def main():
    parser = argparse.ArgumentParser(
        description="Download CLEAN OpenImages V7 data (bounding-box verified only)"
    )
    parser.add_argument("--list-classes", action="store_true",
                        help="List all 600 boxable classes")
    parser.add_argument("--download", action="store_true",
                        help="Download images")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated categories (e.g., 'Apple,Banana,Dog')")
    parser.add_argument("--preset", choices=["diverse", "small"], default=None,
                        help="'diverse' (~70 classes) or 'small' (15 classes)")
    parser.add_argument("--max-per-class", type=int, default=300,
                        help="Max images per class (default: 300)")
    parser.add_argument("--output-dir", type=str, default="./data/openimages_clean",
                        help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Validation split fraction (default: 0.15)")
    args = parser.parse_args()
    
    if args.list_classes:
        list_boxable_classes()
        return
    
    if args.download:
        if args.preset:
            categories = PRESETS[args.preset]
        elif args.categories:
            # Convert simple names to (folder_name, oi_name) tuples
            categories = [
                (c.strip().replace(" ", "_"), c.strip())
                for c in args.categories.split(",")
            ]
        else:
            print("Specify --preset or --categories")
            return
        
        download_with_fiftyone_bbox(
            categories=categories,
            max_per_class=args.max_per_class,
            output_dir=args.output_dir,
            val_split=args.val_split,
        )
        return
    
    parser.print_help()
    print("\n\nQuick start:")
    print("  1. See what's available:")
    print("     python prepare_data_clean.py --list-classes")
    print()
    print("  2. Test with 15 classes:")
    print("     python prepare_data_clean.py --download --preset small --max-per-class 100")
    print()
    print("  3. Full training set:")
    print("     python prepare_data_clean.py --download --preset diverse --max-per-class 500")


if __name__ == "__main__":
    main()
