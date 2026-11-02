import pickle
from hashlib import sha256
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import FeatureUnion


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "aclImdb"
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
BATCH_SIZE = 64


def evaluate_saved_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model artifact not found: {MODEL_PATH}")
    if not VECTORIZER_PATH.is_file():
        raise FileNotFoundError(f"Vectorizer artifact not found: {VECTORIZER_PATH}")

    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with VECTORIZER_PATH.open("rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    if not isinstance(model, LogisticRegression):
        raise TypeError(f"Unexpected model type: {type(model).__name__}")
    if not isinstance(vectorizer, FeatureUnion):
        raise TypeError(f"Unexpected vectorizer type: {type(vectorizer).__name__}")

    labels = []
    predictions = []
    positive_probabilities = []
    independent_labels = []
    independent_predictions = []
    independent_probabilities = []
    training_hashes = set()
    seen_test_reviews = set()
    train_test_overlaps = 0
    duplicate_test_reviews = 0

    for label_name in ("neg", "pos"):
        folder = DATASET_DIR / "train" / label_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Training dataset folder not found: {folder}")

        review_paths = list(folder.glob("*.txt"))
        if not review_paths:
            raise ValueError(f"No training reviews found in {folder}")

        for path in review_paths:
            training_hashes.add(sha256(path.read_bytes()).digest())

    for label_name, target_label in (("neg", 0), ("pos", 1)):
        folder = DATASET_DIR / "test" / label_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Test dataset folder not found: {folder}")

        review_paths = sorted(folder.glob("*.txt"))
        if not review_paths:
            raise ValueError(f"No test reviews found in: {folder}")

        for offset in range(0, len(review_paths), BATCH_SIZE):
            batch_paths = review_paths[offset:offset + BATCH_SIZE]
            raw_reviews = [path.read_bytes() for path in batch_paths]
            texts = [review.decode("utf-8", errors="strict") for review in raw_reviews]
            features = vectorizer.transform(texts)
            batch_predictions = model.predict(features).tolist()
            class_one_index = list(model.classes_).index(1)
            batch_probabilities = model.predict_proba(features)[:, class_one_index].tolist()
            predictions.extend(batch_predictions)
            positive_probabilities.extend(batch_probabilities)
            labels.extend([target_label] * len(texts))

            for raw_review, prediction, probability in zip(
                raw_reviews, batch_predictions, batch_probabilities
            ):
                digest = sha256(raw_review).digest()
                if digest in training_hashes:
                    train_test_overlaps += 1
                    continue

                test_key = (target_label, digest)
                if test_key in seen_test_reviews:
                    duplicate_test_reviews += 1
                    continue

                seen_test_reviews.add(test_key)
                independent_labels.append(target_label)
                independent_predictions.append(prediction)
                independent_probabilities.append(probability)

    accuracy = accuracy_score(labels, predictions)
    print(f"Evaluated {len(labels)} test reviews")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced Accuracy: {balanced_accuracy_score(labels, predictions) * 100:.2f}%")
    print(f"ROC-AUC: {roc_auc_score(labels, positive_probabilities):.4f}")
    print("Confusion matrix (rows=true, columns=predicted; labels=negative, positive):")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))
    print(classification_report(
        labels,
        predictions,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        digits=4,
        zero_division=0,
    ))
    if not independent_labels:
        raise ValueError("No independent, unique test reviews remain after deduplication.")

    independent_accuracy = accuracy_score(
        independent_labels, independent_predictions
    )
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
    print(confusion_matrix(independent_labels, independent_predictions, labels=[0, 1]))
    print(classification_report(
        independent_labels,
        independent_predictions,
        labels=[0, 1],
        target_names=["Negative", "Positive"],
        digits=4,
        zero_division=0,
    ))
    return accuracy


if __name__ == "__main__":
    evaluate_saved_model()
