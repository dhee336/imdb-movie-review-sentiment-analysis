import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
import analyze_dataset
import evaluate_model
import train_model


class DatasetAnalysisTests(unittest.TestCase):
    def test_profile_reports_review_lengths_and_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "first.txt").write_text("A short review", encoding="utf-8")
            (folder / "duplicate.txt").write_text("A short review", encoding="utf-8")
            (folder / "longer.txt").write_text(
                "A much longer movie review", encoding="utf-8"
            )

            stats = analyze_dataset.profile_reviews(folder)

        self.assertEqual(stats["reviews"], 3)
        self.assertEqual(stats["unique_reviews"], 2)
        self.assertAlmostEqual(stats["mean_words"], 11 / 3)
        self.assertEqual(stats["median_words"], 3)
        self.assertEqual(stats["min_words"], 3)
        self.assertEqual(stats["max_words"], 5)

    def test_profile_rejects_missing_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileNotFoundError):
                analyze_dataset.profile_reviews(Path(temp_dir) / "missing")


class TrainModelTests(unittest.TestCase):
    def test_load_reviews_from_missing_folder_reports_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_folder = Path(temp_dir) / "missing"
            with self.assertRaises(FileNotFoundError):
                train_model.load_reviews_from_folder(missing_folder, 1)

    def test_load_reviews_rejects_non_directory_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "not-a-folder"
            file_path.write_text("review", encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                train_model.load_reviews_from_folder(file_path, 1)

    def test_load_reviews_reports_invalid_utf8(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "invalid.txt").write_bytes(b"\xff")
            with self.assertRaises(UnicodeDecodeError):
                train_model.load_reviews_from_folder(folder, 1)

    def test_training_rejects_empty_sentiment_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "aclImdb"
            positive_folder = dataset_dir / "train" / "pos"
            positive_folder.mkdir(parents=True)
            (positive_folder / "review.txt").write_text(
                "A wonderful excellent movie", encoding="utf-8"
            )
            (dataset_dir / "train" / "neg").mkdir()

            with patch.object(train_model, "DATASET_DIR", dataset_dir):
                with redirect_stdout(StringIO()):
                    with self.assertRaisesRegex(ValueError, "negative training reviews"):
                        train_model.train_and_save_model()

    def test_training_rejects_empty_test_sentiment_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "aclImdb"
            for label_name, review in (
                ("pos", "wonderful excellent brilliant movie"),
                ("neg", "awful terrible boring movie"),
            ):
                folder = dataset_dir / "train" / label_name
                folder.mkdir(parents=True)
                (folder / "review.txt").write_text(review, encoding="utf-8")
            (dataset_dir / "test" / "pos").mkdir(parents=True)
            (dataset_dir / "test" / "neg").mkdir()

            with patch.object(train_model, "DATASET_DIR", dataset_dir):
                with redirect_stdout(StringIO()):
                    with self.assertRaisesRegex(ValueError, "No test reviews"):
                        train_model.train_and_save_model()

    def test_training_evaluation_and_saved_artifact_reload(self):
        positive_reviews = [
            "wonderful excellent brilliant story and fantastic acting",
            "excellent wonderful enjoyable movie with brilliant performances",
            "fantastic brilliant story and excellent enjoyable movie",
        ]
        negative_reviews = [
            "awful terrible boring story and horrible acting",
            "terrible awful disappointing movie with boring performances",
            "horrible boring story and terrible disappointing movie",
        ]
        positive_test_reviews = [
            positive_reviews[0],
            "A delightful uplifting film with charming performances and a moving story",
        ]
        negative_test_reviews = [
            negative_reviews[0],
            "A dreadful tedious film with a miserable plot and irritating acting",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            dataset_dir = temp_root / "aclImdb"
            for split, reviews_per_label in (
                ("train", (positive_reviews, negative_reviews)),
                ("test", (positive_test_reviews, negative_test_reviews)),
            ):
                for label_name, reviews in zip(("pos", "neg"), reviews_per_label):
                    folder = dataset_dir / split / label_name
                    folder.mkdir(parents=True)
                    for index, review in enumerate(reviews):
                        (folder / f"{index}.txt").write_text(review, encoding="utf-8")

            model_path = temp_root / "model.pkl"
            vectorizer_path = temp_root / "vectorizer.pkl"
            with patch.multiple(
                train_model,
                DATASET_DIR=dataset_dir,
                MODEL_PATH=model_path,
                VECTORIZER_PATH=vectorizer_path,
            ), redirect_stdout(StringIO()) as output:
                train_model.train_and_save_model()

            self.assertIn("Test Accuracy:", output.getvalue())
            self.assertIn("Balanced Accuracy:", output.getvalue())
            self.assertIn("ROC-AUC:", output.getvalue())
            self.assertIn("Confusion matrix", output.getvalue())
            self.assertIn("excluded 2 train/test overlaps", output.getvalue())
            self.assertIn("Leakage-free Test Accuracy:", output.getvalue())
            self.assertTrue(model_path.is_file())
            self.assertTrue(vectorizer_path.is_file())

            with model_path.open("rb") as model_file:
                model = pickle.load(model_file)
            with vectorizer_path.open("rb") as vectorizer_file:
                vectorizer = pickle.load(vectorizer_file)

            self.assertIsInstance(model, LogisticRegression)
            self.assertIsInstance(vectorizer, FeatureUnion)
            self.assertEqual(
                int(model.predict(vectorizer.transform([positive_reviews[0]]))[0]),
                1,
            )
            self.assertEqual(
                int(model.predict(vectorizer.transform([negative_reviews[0]]))[0]),
                0,
            )

            with patch.multiple(
                evaluate_model,
                DATASET_DIR=dataset_dir,
                MODEL_PATH=model_path,
                VECTORIZER_PATH=vectorizer_path,
                BATCH_SIZE=1,
            ), redirect_stdout(StringIO()) as evaluation_output:
                accuracy = evaluate_model.evaluate_saved_model()

            self.assertGreaterEqual(accuracy, 0.5)
            self.assertIn("Evaluated 4 test reviews", evaluation_output.getvalue())
            self.assertIn("Balanced Accuracy:", evaluation_output.getvalue())
            self.assertIn("ROC-AUC:", evaluation_output.getvalue())
            self.assertIn("Confusion matrix", evaluation_output.getvalue())
            self.assertIn("Leakage-free unique test reviews: 2", evaluation_output.getvalue())
            self.assertIn("excluded 2 train/test overlaps", evaluation_output.getvalue())


if __name__ == "__main__":
    unittest.main()
