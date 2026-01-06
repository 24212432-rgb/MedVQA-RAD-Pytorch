import json
import random
import argparse
from collections import defaultdict
from pathlib import Path


def load_data(json_path):
    """Load VQA-RAD JSON file"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} QA pairs from {json_path}")
    return data


def group_by_image(data):
    """Group all QA pairs by their image"""
    image_to_qa = defaultdict(list)
    
    for item in data:
        # Try different possible field names for image
        img_name = item.get('image_name', item.get('image', item.get('img_name', '')))
        if img_name:
            image_to_qa[img_name].append(item)
    
    print(f"Found {len(image_to_qa)} unique images")
    return image_to_qa


def analyze_questions(qa_list):
    """Analyze question types in a list of QA pairs"""
    close_count = 0
    open_count = 0
    
    for item in qa_list:
        answer = str(item.get('answer', '')).lower().strip()
        if answer in ['yes', 'no']:
            close_count += 1
        else:
            open_count += 1
    
    return close_count, open_count


def create_image_disjoint_split(image_to_qa, test_ratio=0.2, seed=42):
    """
    Create train/test split where NO IMAGE appears in both sets.
    
    Strategy:
    1. Group all QA pairs by image
    2. Randomly assign entire images (with all their QA pairs) to train or test
    3. Try to balance closed/open question distribution
    """
    random.seed(seed)
    
    # Get all image names
    all_images = list(image_to_qa.keys())
    random.shuffle(all_images)
    
    # Calculate target test size
    total_qa = sum(len(qa_list) for qa_list in image_to_qa.values())
    target_test_qa = int(total_qa * test_ratio)
    
    print(f"\nTarget split: ~{100*(1-test_ratio):.0f}% train, ~{100*test_ratio:.0f}% test")
    print(f"Total QA pairs: {total_qa}, Target test QA: ~{target_test_qa}")
    
    # Assign images to train/test
    test_images = []
    train_images = []
    test_qa_count = 0
    
    for img in all_images:
        qa_count = len(image_to_qa[img])
        
        if test_qa_count < target_test_qa:
            test_images.append(img)
            test_qa_count += qa_count
        else:
            train_images.append(img)
    
    # Build train and test sets
    train_data = []
    test_data = []
    
    for img in train_images:
        train_data.extend(image_to_qa[img])
    
    for img in test_images:
        test_data.extend(image_to_qa[img])
    
    # Shuffle within each set
    random.shuffle(train_data)
    random.shuffle(test_data)
    
    return train_data, test_data, set(train_images), set(test_images)


def verify_no_leakage(train_images, test_images):
    """Verify there is NO image overlap between train and test"""
    overlap = train_images & test_images
    
    if len(overlap) == 0:
        print("\n" + "=" * 60)
        print(" VERIFICATION PASSED: NO IMAGE LEAKAGE!")
        print("=" * 60)
        print(f"  Train images: {len(train_images)}")
        print(f"  Test images:  {len(test_images)}")
        print(f"  Overlap:      0 (PERFECT!)")
        print("=" * 60)
        return True
    else:
        print("\n" + "=" * 60)
        print(" WARNING: IMAGE LEAKAGE DETECTED!")
        print("=" * 60)
        print(f"  Overlapping images: {len(overlap)}")
        print("=" * 60)
        return False


def print_statistics(train_data, test_data, train_images, test_images):
    """Print detailed statistics"""
    train_close, train_open = analyze_questions(train_data)
    test_close, test_open = analyze_questions(test_data)
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS (Image-Disjoint Split)")
    print("=" * 60)
    
    print(f"\n{'Split':<10} {'QA Pairs':<12} {'Images':<10} {'Closed':<12} {'Open':<12}")
    print("-" * 60)
    print(f"{'Train':<10} {len(train_data):<12} {len(train_images):<10} {train_close:<12} {train_open:<12}")
    print(f"{'Test':<10} {len(test_data):<12} {len(test_images):<10} {test_close:<12} {test_open:<12}")
    print(f"{'Total':<10} {len(train_data)+len(test_data):<12} {len(train_images)+len(test_images):<10} {train_close+test_close:<12} {train_open+test_open:<12}")
    
    print(f"\nQuestion Type Distribution:")
    print(f"  Train: {100*train_close/(train_close+train_open):.1f}% closed, {100*train_open/(train_close+train_open):.1f}% open")
    print(f"  Test:  {100*test_close/(test_close+test_open):.1f}% closed, {100*test_open/(test_close+test_open):.1f}% open")
    
    print("=" * 60)


def save_splits(train_data, test_data, output_dir):
    """Save train and test splits to JSON files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / "trainset_image_disjoint.json"
    test_path = output_dir / "testset_image_disjoint.json"
    
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n Saved train set: {train_path} ({len(train_data)} samples)")
    print(f" Saved test set:  {test_path} ({len(test_data)} samples)")
  
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Create image-disjoint VQA-RAD split")
    parser.add_argument("--input", type=str, default="VQA_RAD Dataset Public.json",
                        help="Path to original VQA-RAD JSON file")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory for split files")
    parser.add_argument("--test_ratio", type=float, default=0.2,
                        help="Fraction of data for test set (default: 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("VQA-RAD IMAGE-DISJOINT DATA SPLIT")
    print("Eliminating data leakage for fair evaluation")
    print("=" * 60)
    
    # Load data
    data = load_data(args.input)
    
    # Group by image
    image_to_qa = group_by_image(data)
    
    # Create split
    train_data, test_data, train_images, test_images = create_image_disjoint_split(
        image_to_qa, 
        test_ratio=args.test_ratio, 
        seed=args.seed
    )
    
    # Verify no leakage
    is_valid = verify_no_leakage(train_images, test_images)
    
    if not is_valid:
        print("ERROR: Split verification failed!")
        return
    
    # Print statistics
    print_statistics(train_data, test_data, train_images, test_images)
    
    # Save
    save_splits(train_data, test_data, args.output_dir)
    
    print("\n" + "=" * 60)
    print("DONE! Use these files for training:")
    print(f"  Train: {args.output_dir}/trainset_image_disjoint.json")
    print(f"  Test:  {args.output_dir}/testset_image_disjoint.json")
    print("=" * 60)


if __name__ == "__main__":
    main()