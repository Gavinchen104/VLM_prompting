"""
01_download_data.py — Download HAM10000 and build a balanced pilot subset.

Run on the LOGIN NODE (it has internet). The compute nodes may not.

What this does:
1. Downloads HAM10000 from Kaggle into ./data/ham10000/
2. Reads the metadata CSV
3. Picks 3 images per class (21 total) for the pilot.
   - We deliberately mix easy and tricky cases by picking via fixed random seed.
4. Writes pilot_manifest.csv with columns: image_id, true_label, image_path

After this runs you should see:
    data/ham10000/HAM10000_metadata.csv
    data/ham10000/HAM10000_images_part_1/  (and part_2)
    pilot_manifest.csv  (21 rows)
"""

import os
import sys
import zipfile
import shutil
import subprocess
from pathlib import Path
import pandas as pd

# ---------- Config ----------
SEED = 42
PER_CLASS = 3                      # 3 per class × 7 classes = 21 pilot images
DATA_DIR = Path("data/ham10000")
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"
MANIFEST = Path("pilot_manifest.csv")


def run(cmd, **kwargs):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def download():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if (DATA_DIR / "HAM10000_metadata.csv").exists():
        print("HAM10000 already downloaded — skipping.")
        return

    print("Downloading HAM10000 from Kaggle (~3GB) ...")
    # Will download a zip into DATA_DIR
    run(f"kaggle datasets download -d {KAGGLE_DATASET} -p {DATA_DIR} --unzip")

    # Sanity check
    if not (DATA_DIR / "HAM10000_metadata.csv").exists():
        print("ERROR: metadata file not found after download. Listing contents:")
        for p in DATA_DIR.rglob("*"):
            print(" ", p)
        sys.exit(1)

    print("Download complete.")


def find_image_path(image_id: str) -> Path | None:
    """HAM10000 ships images in two folders: part_1 and part_2. Look in both."""
    for sub in ["HAM10000_images_part_1", "HAM10000_images_part_2",
                "ham10000_images_part_1", "ham10000_images_part_2"]:
        cand = DATA_DIR / sub / f"{image_id}.jpg"
        if cand.exists():
            return cand
    # fallback: search
    hits = list(DATA_DIR.rglob(f"{image_id}.jpg"))
    return hits[0] if hits else None


def build_pilot_manifest():
    meta_path = DATA_DIR / "HAM10000_metadata.csv"
    df = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(df)} rows")
    print("Class distribution:")
    print(df["dx"].value_counts())

    # Sample PER_CLASS images per class with a fixed seed.
    # Use groupby().sample() — pandas 3.0 excludes the grouping column from
    # groupby().apply() by default, which dropped 'dx' from the result.
    sampled = (df.groupby("dx", group_keys=False)
                 .sample(n=PER_CLASS, random_state=SEED)
                 .reset_index(drop=True))
    print(f"\nSampled {len(sampled)} images for the pilot.")

    rows = []
    for _, row in sampled.iterrows():
        path = find_image_path(row["image_id"])
        if path is None:
            print(f"WARNING: image not found for {row['image_id']}")
            continue
        rows.append({
            "image_id": row["image_id"],
            "true_label": row["dx"],
            "image_path": str(path.resolve()),
        })

    out = pd.DataFrame(rows)
    out.to_csv(MANIFEST, index=False)
    print(f"\nWrote {MANIFEST} ({len(out)} rows)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    download()
    build_pilot_manifest()
