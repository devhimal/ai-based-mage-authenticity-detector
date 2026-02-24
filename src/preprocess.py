# src/preprocess.py
import os
import shutil
import random
import cv2
from tqdm import tqdm

def preprocess_images(image_size=(96, 96), split_ratios=(0.7, 0.15, 0.15), max_images_per_class=None):
    """
    Processes raw images and splits them into train, validation, and test sets.
    - Reads images from `data/raw`.
    - Resizes them to a uniform size.
    - Saves them into `data/processed/{train|validation|test}/{real|fake}`.
    """
    print("Starting image preprocessing and splitting...")

    base_dir = os.path.join(os.path.dirname(__file__), '..')
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')

    if not os.path.exists(raw_data_dir):
        print("Error: Raw data directory not found. Please run download_data.py first.")
        return

    # Clean up previous processed data if it exists
    if os.path.exists(processed_data_dir):
        print("Removing existing processed data directory...")
        shutil.rmtree(processed_data_dir)

    print(f"Creating new processed data directory at: {processed_data_dir}")

    # Create directory structure: data/processed/{train|validation|test}/{real|fake}
    split_names = ['train', 'validation', 'test']
    class_names = ['real', 'fake']
    for split_name in split_names:
        for class_name in class_names:
            os.makedirs(os.path.join(processed_data_dir, split_name, class_name), exist_ok=True)

    # --- Process and Split Logic ---
    train_ratio, val_ratio, test_ratio = split_ratios

    for class_name in class_names:
        print(f"\nProcessing class: {class_name}")
        source_dir = os.path.join(raw_data_dir, class_name)
        
        if not os.path.isdir(source_dir):
            print(f"Warning: Source directory not found for class '{class_name}'. Skipping.")
            continue

        images = [f for f in os.listdir(source_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)

        if max_images_per_class is not None:
            images = images[:max_images_per_class]
            print(f"Limiting to {max_images_per_class} images for class '{class_name}'.")

        # Calculate split indices
        train_end = int(len(images) * train_ratio)
        val_end = train_end + int(len(images) * val_ratio)

        splits = {
            'train': images[:train_end],
            'validation': images[train_end:val_end],
            'test': images[val_end:]
        }

        # Process and copy files
        for split_name, image_list in splits.items():
            print(f"Processing {len(image_list)} images for {split_name} set...")
            dest_dir = os.path.join(processed_data_dir, split_name, class_name)
            
            for image_name in tqdm(image_list, desc=f"  {split_name} ({class_name})"):
                source_path = os.path.join(source_dir, image_name)
                dest_path = os.path.join(dest_dir, image_name)

                try:
                    # Read, resize, and save the image
                    img = cv2.imread(source_path)
                    if img is None:
                        print(f"Warning: Could not read image {source_path}. Skipping.")
                        continue
                    
                    img_resized = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(dest_path, img_resized)

                except Exception as e:
                    print(f"Error processing image {source_path}: {e}")

    print("\nImage preprocessing and splitting complete.")
    print(f"Processed data is located in: {processed_data_dir}")

if __name__ == "__main__":
    # To train on a smaller subset of data, you can specify the max_images_per_class parameter.
    # For example, to use 1000 images from each class (real and fake):
    preprocess_images(max_images_per_class=7038)