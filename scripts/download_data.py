"""
Optional helper: fetch the public Kaggle dataset with kagglehub and copy the
CSV to data/house_sale.csv. Equivalent to downloading it by hand (see
data/README.md).

    python scripts/download_data.py
"""

import os
import shutil

import kagglehub

DATASET = "sehriyarmemmedli/binaaz-sale-project"
TARGET = os.path.join("data", "house_sale.csv")

if __name__ == "__main__":
    path = kagglehub.dataset_download(DATASET)
    src = os.path.join(path, "house_sale.csv")
    os.makedirs("data", exist_ok=True)
    shutil.copyfile(src, TARGET)
    print(f"Copied {src} -> {TARGET} ({os.path.getsize(TARGET) / 1e6:.1f} MB)")
