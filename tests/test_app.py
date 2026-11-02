import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SentimentAppTests(unittest.TestCase):
    def test_review_validation_and_predictions(self):
        app = AppTest.from_file(str(PROJECT_ROOT / "app.py")).run(timeout=60)
        self.assertFalse(app.exception, [str(error.value) for error in app.exception])
        self.assertIn("Movie Review Sentiment Analysis", app.title[0].value)

        app.button[0].click().run(timeout=60)
        self.assertTrue(any("Please enter some text" in item.value for item in app.warning))

        app.text_area[0].set_value("This movie was really very good").run(timeout=60)
        app.button[0].click().run(timeout=60)
        self.assertTrue(any("at least 8 words" in item.value for item in app.warning))

        app.text_area[0].set_value(
            "This excellent and wonderful movie has a brilliant story and fantastic acting."
        ).run(timeout=60)
        app.button[0].click().run(timeout=60)
        self.assertTrue(any("POSITIVE REVIEW" in item.value for item in app.success))

        app.text_area[0].set_value(
            "This ggod movie has excellent acting and a wonderful story."
        ).run(timeout=60)
        app.button[0].click().run(timeout=60)
        self.assertTrue(
            any("Spelling corrections applied: This good movie" in item.value
                for item in app.info)
        )
        self.assertTrue(any("POSITIVE REVIEW" in item.value for item in app.success))

        app.text_area[0].set_value(
            "The movie is good bandit and good plot."
        ).run(timeout=60)
        app.button[0].click().run(timeout=60)
        self.assertFalse(
            any("Spelling corrections applied" in item.value for item in app.info)
        )

        review_with_single_typo = (
            "The movie is ggod and it is ok movie. The twist is unexpected."
        )
        app.text_area[0].set_value(review_with_single_typo).run(timeout=60)
        app.button[0].click().run(timeout=60)
        self.assertTrue(
            any(
                "Spelling corrections applied: The movie is good and it is ok movie."
                in item.value
                for item in app.info
            )
        )

        app.text_area[0].set_value(
            "This awful and boring movie has a terrible story and horrible acting."
        ).run(timeout=60)
        app.button[0].click().run(timeout=60)
        self.assertTrue(any("NEGATIVE REVIEW" in item.value for item in app.error))

        app.text_area[0].set_value(
            "A person walks into a room. Another person speaks to him. "
            "They leave the room and walk down the street. Later they return home."
        ).run(timeout=60)
        app.button[0].click().run(timeout=60)
        self.assertTrue(any("UNCLEAR REVIEW" in item.value for item in app.warning))


if __name__ == "__main__":
    unittest.main()
