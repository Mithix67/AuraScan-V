import os
import numpy as np
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────────
IMG_SIZE = 256
PROCESSED_DIR = "data/processed"

def generate_mock_data(num_samples=100):
    """
    Generates synthetic data for testing the pipeline WITHOUT real patient data.

    Creates 2D grayscale NumPy arrays simulating CT slices:
        - Benign (label=0): Random Gaussian noise background.
        - Malignant (label=1): Gaussian noise + a bright circular "nodule" region.

    Parameters:
        num_samples (int): Number of synthetic samples to generate.

    Output:
        Saves .npy files to PROCESSED_DIR, each containing {'image': np.ndarray, 'label': int}.
    """
    print(f"Generating {num_samples} mock samples in {PROCESSED_DIR}...")

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    for i in tqdm(range(num_samples)):
        # Create a blank image
        image = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

        # Add Gaussian noise (simulates CT scanner noise)
        noise = np.random.normal(0, 0.1, (IMG_SIZE, IMG_SIZE))
        image += noise

        # Randomly assign label (0: Benign, 1: Malignant)
        label = np.random.randint(0, 2)

        # If malignant, add a synthetic "nodule" (bright circular spot)
        if label == 1:
            cx, cy = np.random.randint(50, 200), np.random.randint(50, 200)
            y, x = np.ogrid[-cx:IMG_SIZE-cx, -cy:IMG_SIZE-cy]
            mask = x*x + y*y <= 20*20
            image[mask] = 1.0  # Nodule represented as max intensity

        # Normalize to [0, 1]
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        sample = {'image': image, 'label': label}
        np.save(os.path.join(PROCESSED_DIR, f'sample_{i}.npy'), sample)

    print(f"Done. {num_samples} samples saved to '{PROCESSED_DIR}'.")

if __name__ == "__main__":
    generate_mock_data()
