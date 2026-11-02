import pickle
from hashlib import sha256
from pathlib import Path
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "aclImdb"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


def load_reviews_from_folder(folder_path, label):
    texts = []
    labels = []

    if not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for file_path in sorted(folder_path.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        texts.append(text)
        labels.append(label)

    return texts, labels


def train_and_save_model():
    print("Loading training dataset...")

    texts = []
    labels = []

    pos_texts, pos_labels = load_reviews_from_folder(
        DATASET_DIR / "train" / "pos", 1
    )
    texts.extend(pos_texts)
    labels.extend(pos_labels)

    neg_texts, neg_labels = load_reviews_from_folder(
        DATASET_DIR / "train" / "neg", 0
    )
    texts.extend(neg_texts)
    labels.extend(neg_labels)

    if not pos_texts:
        raise ValueError(
            f"No positive training reviews found in {DATASET_DIR / 'train' / 'pos'}"
        )
    if not neg_texts:
        raise ValueError(
            f"No negative training reviews found in {DATASET_DIR / 'train' / 'neg'}"
        )

    for label_name in ("pos", "neg"):
        folder = DATASET_DIR / "test" / label_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {folder}")
        if not any(folder.glob("*.txt")):
            raise ValueError(f"No test reviews found in {folder}")

    print(f"Loaded {len(texts)} training reviews")
    print(f"Positive: {len(pos_labels)}")
    print(f"Negative: {len(neg_labels)}")

    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(
            stop_words=None,
            sublinear_tf=True,
            ngram_range=(1, 2),
            min_df=2
        )),
        ("character", TfidfVectorizer(
            analyzer="char",
            sublinear_tf=True,
            ngram_range=(3, 5),
            min_df=3
        ))
    ])

    x_train = vectorizer.fit_transform(texts)

    print(f"Created {x_train.shape[1]} TF-IDF features")

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, labels)

    print("Model training complete")

    print("Loading test dataset...")

    test_texts = []
    test_labels = []

    test_pos_texts, test_pos_labels = load_reviews_from_folder(
        DATASET_DIR / "test" / "pos", 1
    )
    test_texts.extend(test_pos_texts)
    test_labels.extend(test_pos_labels)

    test_neg_texts, test_neg_labels = load_reviews_from_folder(
        DATASET_DIR / "test" / "neg", 0
    )
    test_texts.extend(test_neg_texts)
    test_labels.extend(test_neg_labels)

    if not test_pos_texts or not test_neg_texts:
        raise ValueError("The test dataset must contain positive and negative reviews.")

    print(f"Loaded {len(test_texts)} test reviews")

    x_test = vectorizer.transform(test_texts)
    predictions = model.predict(x_test)
    class_one_index = list(model.classes_).index(1)
    positive_probabilities = model.predict_proba(x_test)[:, class_one_index]

    accuracy = accuracy_score(test_labels, predictions)
    training_hashes = {
        sha256(text.encode("utf-8")).digest()
        for text in texts
    }
    seen_test_reviews = set()
    independent_test_indices = []
    train_test_overlaps = 0
    duplicate_test_reviews = 0

    for index, (text, label) in enumerate(zip(test_texts, test_labels)):
        digest = sha256(text.encode("utf-8")).digest()
        if digest in training_hashes:
            train_test_overlaps += 1
            continue

        test_key = (label, digest)
        if test_key in seen_test_reviews:
            duplicate_test_reviews += 1
            continue

        seen_test_reviews.add(test_key)
        independent_test_indices.append(index)

    if not independent_test_indices:
        raise ValueError("No independent, unique test reviews remain after deduplication.")

    independent_labels = [test_labels[index] for index in independent_test_indices]
    independent_predictions = [predictions[index] for index in independent_test_indices]
    independent_probabilities = [
        positive_probabilities[index] for index in independent_test_indices
    ]
    independent_accuracy = accuracy_score(
        independent_labels, independent_predictions
    )

    print()
    print("=" * 50)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy_score(test_labels, predictions) * 100:.2f}%"
    )
    print(f"ROC-AUC: {roc_auc_score(test_labels, positive_probabilities):.4f}")
    print("Confusion matrix (rows=true, columns=predicted; labels=negative, positive):")
    print(confusion_matrix(test_labels, predictions, labels=[0, 1]))
    print("=" * 50)
    print()
    print(classification_report(
        test_labels,
        predictions,
        target_names=["Negative", "Positive"],
        zero_division=0,
    ))
    print(
        f"Leakage-free unique test reviews: {len(independent_labels)} "
        f"(excluded {train_test_overlaps} train/test overlaps and "
        f"{duplicate_test_reviews} repeated test reviews)"
    )
    print(f"Leakage-free Test Accuracy: {independent_accuracy * 100:.2f}%")
    print(
        "Leakage-free Balanced Accuracy: "
        f"{balanced_accuracy_score(independent_labels, independent_predictions) * 100:.2f}%"
    )
    print(
        "Leakage-free ROC-AUC: "
        f"{roc_auc_score(independent_labels, independent_probabilities):.4f}"
    )
    print("Leakage-free confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(
        independent_labels, independent_predictions, labels=[0, 1]
    ))
    print(classification_report(
        independent_labels,
        independent_predictions,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        zero_division=0,
    ))

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    with VECTORIZER_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print("Training and evaluation complete")


if __name__ == "__main__":
    train_and_save_model()