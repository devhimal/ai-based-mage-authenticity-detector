import os
import requests
import zipfile
import tarfile
from tqdm import tqdm
import subprocess # Import subprocess

def download_file(url, filename):
    """Helper function to download a file with a progress bar."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size_in_bytes = int(r.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(filename, 'wb') as f:
            with tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(filename)}") as pbar:
                for chunk in r.iter_content(chunk_size=block_size):
                    pbar.update(len(chunk))
                    f.write(chunk)


def download_and_extract_data():
    """
    Downloads and extracts the datasets.
    - Real images: A subset of CelebA (Align&Cropped)
    - Fake images: A subset of "This Person Does Not Exist" (TPDNE) images from a pre-packaged source.
    """
    print("Starting data download and extraction...")
    
    # Create a directory for the datasets
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    raw_data_dir = os.path.join(data_dir, 'raw')
    os.makedirs(raw_data_dir, exist_ok=True)

    # --- Dataset IDs for Kaggle ---
    real_dataset_id = 'jessicali9530/celeba-dataset'
    fake_dataset_id = 'ashishpatel26/person-face-dataset-thispersondoesnotexist'

    real_zip_name = 'celeba-dataset.zip' # This is the name kaggle downloads it as
    real_zip_path = os.path.join(raw_data_dir, real_zip_name)
    real_extract_path = os.path.join(raw_data_dir, 'real')

    fake_zip_name = 'images.zip' # Manually downloaded file
    fake_zip_path = os.path.join(raw_data_dir, fake_zip_name)
    fake_extract_path = os.path.join(raw_data_dir, 'fake')


    # --- Download and Extract Real Images ---
    if not os.path.exists(real_extract_path):
        print(f"\nDownloading Real images (CelebA subset) from Kaggle: {real_dataset_id}...")
        subprocess.run(["kaggle", "datasets", "download", "-d", real_dataset_id, "-p", raw_data_dir], check=True)
        print(f"Extracting {real_zip_path}...")
        with zipfile.ZipFile(real_zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_data_dir)
            # Kaggle's CelebA usually extracts to 'img_align_celeba'
            extracted_folder = os.path.join(raw_data_dir, 'img_align_celeba')
            if os.path.exists(extracted_folder):
                os.rename(extracted_folder, real_extract_path)
        os.remove(real_zip_path)
        print("Real images downloaded and extracted.")
    else:
        print("Real images directory already exists. Skipping download.")

    # --- Extract Fake Images (Assuming manual download) ---
    if not os.path.exists(fake_extract_path):
        print("\nExtracting Fake images (TPDNE subset) from local file...")
        if os.path.exists(fake_zip_path):
            with zipfile.ZipFile(fake_zip_path, 'r') as zip_ref:
                zip_ref.extractall(raw_data_dir)
                extracted_folder = os.path.join(raw_data_dir, 'images') # Kaggle TPDNE extracts to 'images'
                if os.path.exists(extracted_folder):
                    os.rename(extracted_folder, fake_extract_path)
            os.remove(fake_zip_path)
            print("Fake images extracted.")
        else:
            print(f"Error: TPDNE dataset not found at {fake_zip_path}. Please download it manually and place it there.")
            print("Download from: https://www.kaggle.com/datasets/ashishpatel26/person-face-dataset-thispersondoesnotexist")
    else:
        print("Fake images directory already exists. Skipping extraction.")

    print("\nData download and extraction process complete.")
    print(f"Data is located in: {raw_data_dir}")


if __name__ == "__main__":
    download_and_extract_data()