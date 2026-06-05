"""Download Quick Draw bitmap data for the chosen categories and merge into X/y arrays."""
import os
import subprocess
import urllib.parse
import numpy as np

from categories import CATEGORIES

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
BASE_URL = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
PER_CLASS = 30_000


def download_category(name: str) -> str:
    """Download a single category .npy file if not already cached. Returns local path."""
    fname = f"{name}.npy"
    local_path = os.path.join(DATA_DIR, fname)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        print(f"  cached: {fname}")
        return local_path

    url = f"{BASE_URL}/{urllib.parse.quote(name)}.npy"
    print(f"  downloading: {url}")
    # Use curl: macOS Python from python.org has no system CA bundle.
    subprocess.run(
        ["curl", "-fL", "-sS", "-o", local_path, url],
        check=True,
    )
    return local_path


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    X_chunks = []
    y_chunks = []
    for class_idx, name in enumerate(CATEGORIES):
        print(f"[{class_idx + 1}/{len(CATEGORIES)}] {name}")
        path = download_category(name)
        arr = np.load(path)
        arr = arr[:PER_CLASS]
        print(f"  shape: {arr.shape}")
        X_chunks.append(arr)
        y_chunks.append(np.full(arr.shape[0], class_idx, dtype=np.int64))

    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    print(f"\nMerged: X={X.shape}, y={y.shape}")

    np.save(os.path.join(DATA_DIR, "X.npy"), X)
    np.save(os.path.join(DATA_DIR, "y.npy"), y)
    print(f"Saved X.npy and y.npy in {DATA_DIR}")


if __name__ == "__main__":
    main()
