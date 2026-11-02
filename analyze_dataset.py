"""Print reproducible dataset statistics for the IMDb review folders."""

from hashlib import sha256
from pathlib import Path
from statistics import mean, median

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "aclImdb"
SPLITS = ("train", "test")
LABELS = ("neg", "pos")


def profile_reviews(folder_path):
    if not folder_path.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {folder_path}")

    review_paths = sorted(folder_path.glob("*.txt"))
    if not review_paths:
        raise ValueError(f"No review text files found in {folder_path}")

    word_counts = []
    review_hashes = set()
    for path in review_paths:
        text = path.read_text(encoding="utf-8", errors="strict")
        word_counts.append(len(text.split()))
        review_hashes.add(sha256(text.encode("utf-8")).digest())

    return {
        "reviews": len(review_paths),
        "unique_reviews": len(review_hashes),
        "mean_words": mean(word_counts),
        "median_words": median(word_counts),
        "min_words": min(word_counts),
        "max_words": max(word_counts),
    }


def report_dataset(dataset_dir=DATASET_DIR):
    print("IMDb dataset profile")
    print("=" * 72)
    for split in SPLITS:
        for label in LABELS:
            folder = dataset_dir / split / label
            stats = profile_reviews(folder)
            print(
                f"{split:5} {label:3} | "
                f"{stats['reviews']:>6} reviews | "
                f"{stats['unique_reviews']:>6} unique | "
                f"words mean/median: {stats['mean_words']:.1f}/"
                f"{stats['median_words']:.1f} | "
                f"range: {stats['min_words']}-{stats['max_words']}"
            )


if __name__ == "__main__":
    report_dataset()
