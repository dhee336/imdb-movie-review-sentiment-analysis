import pickle
import re
from pathlib import Path
from spellchecker import SpellChecker
import streamlit as st
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
DATASET_DIR = BASE_DIR / "aclImdb"
MIN_CONFIDENCE = 0.60
COMMON_CORRECTIONS = {
    "fanatsti": "fantastic",
    "fantastci": "fantastic",
    "amzaing": "amazing",
    "amazng": "amazing",
    "excellnt": "excellent",
    "borng": "boring",
}
SPELL_CHECKER = SpellChecker()
POSITIVE_WORDS = {
    "amazing", "awesome", "beautiful", "best", "brilliant", "enjoyed",
    "excellent", "fantastic", "good", "great", "interesting", "love",
    "loved", "perfect", "wonderful", "entertaining",
}
NEGATIVE_WORDS = {
    "awful", "bad", "boring", "disappointing", "hate", "hated", "horrible",
    "poor", "terrible", "waste", "worst", "unwatchable",
}


def correct_spelling(text):
    def correct_word(match):
        word = match.group(0)
        if len(word) < 3:
            return word

        normalized_word = word.lower()
        if normalized_word in SPELL_CHECKER:
            return word

        correction = SPELL_CHECKER.correction(normalized_word)
        if correction is None:
            return word
        if word.isupper():
            return correction.upper()
        if word.istitle():
            return correction.capitalize()
        return correction

    return re.sub(r"\b[A-Za-z]+\b", correct_word, text)


def normalize_review(review_text):
    review_text = re.sub(r"\bnot\s+really\s+a\s+good\s+movie\b", "bad movie", review_text, flags=re.IGNORECASE)
    review_text = re.sub(r"\bnot\s+a\s+good\s+movie\b", "bad movie", review_text, flags=re.IGNORECASE)
    review_text = re.sub(r"\bless\s+tham\b", "less than", review_text, flags=re.IGNORECASE)
    review_text = re.sub(r"\bless\s+than\s+one\s+time\s+watch\b", "not worth watching", review_text, flags=re.IGNORECASE)
    review_text = re.sub(r"\bnot\s+boring\b", "entertaining", review_text, flags=re.IGNORECASE)
    review_text = re.sub(r"\bnot\s+bad\b", "good", review_text, flags=re.IGNORECASE)
    review_text = re.sub(r"\bnot\s+awful\b", "good", review_text, flags=re.IGNORECASE)
    review_text = correct_spelling(review_text)
    words = re.findall(r"\b[\w']+\b|[^\w\s]", review_text)
    return " ".join(COMMON_CORRECTIONS.get(word.lower(), word) for word in words)


def apply_clear_sentiment(review_text, prediction):
    words = set(re.findall(r"\b[\w']+\b", review_text.lower()))
    positive_score = len(words & POSITIVE_WORDS)
    negative_score = len(words & NEGATIVE_WORDS)

    if positive_score >= 2 and negative_score == 0:
        return 1
    if negative_score >= 2 and positive_score == 0:
        return 0
    return prediction


def train_model_from_scratch():
    print("Training new model...")

    texts = []
    labels = []

    for label_name, target_label in [("pos", 1), ("neg", 0)]:
        folder = DATASET_DIR / "train" / label_name

        if not folder.is_dir():
            raise FileNotFoundError(f"Dataset not found: {folder}")

        review_count = 0
        for file_path in sorted(folder.glob("*.txt")):
            text = file_path.read_text(encoding="utf-8")
            texts.append(text)
            labels.append(target_label)
            review_count += 1

        if not review_count:
            sentiment = "positive" if target_label == 1 else "negative"
            raise ValueError(f"No {sentiment} training reviews found in {folder}")

    if not texts:
        raise ValueError("No training data found!")

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

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, labels)

    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    with VECTORIZER_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)

    print("Model training complete!")
    return model, vectorizer


def load_model_and_vectorizer():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return train_model_from_scratch()

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)

    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)

    if not isinstance(model, LogisticRegression) or not isinstance(vectorizer, FeatureUnion):
        return train_model_from_scratch()

    return model, vectorizer


def predict_sentiment(review_text, model, vectorizer):
    normalized_text = normalize_review(review_text)
    vector = vectorizer.transform([normalized_text])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    prediction = apply_clear_sentiment(normalized_text, prediction)
    confidence = probabilities[list(model.classes_).index(prediction)]
    return prediction, confidence


st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

model, vectorizer = load_model_and_vectorizer()

st.title("🎬 Movie Review Sentiment Analysis")
st.caption("An interpretable NLP classification demo using IMDb movie reviews.")

st.subheader("Predict review sentiment")
st.write("Enter an English-language movie review to classify it as positive or negative.")
review = st.text_area(
    "📝 Enter your movie review:",
    height=150,
    placeholder="Example: This movie was absolutely amazing! The acting was wonderful..."
)

if st.button(
    "🔮 Predict Sentiment",
    type="primary",
    width="stretch"
):
    if not review.strip():
        st.warning("⚠️ Please enter some text before predicting.")
    elif len(review.split()) < 8:
        st.warning(
            "⚠️ Please enter a longer review (at least 8 words) "
            "so the model has enough information."
        )
    else:
        corrected_review = correct_spelling(review)
        if corrected_review != review:
            st.info(f"Spelling corrections applied: {corrected_review}")

        prediction, confidence = predict_sentiment(review, model, vectorizer)
        confidence_percent = confidence * 100

        if confidence < MIN_CONFIDENCE:
            st.warning("⚖️ **UNCLEAR REVIEW**")
            st.write(
                f"The model is not confident enough "
                f"({confidence_percent:.1f}% estimated probability)."
            )
        elif prediction == 1:
            st.success(f"✅ **POSITIVE REVIEW** ({confidence_percent:.1f}% estimated probability)")
            st.write("The model predicts this is a positive review.")
        else:
            st.error(f"❌ **NEGATIVE REVIEW** ({confidence_percent:.1f}% estimated probability)")
            st.write("The model predicts this is a negative review.")

st.caption(
    "Estimated probabilities are model scores, not calibrated guarantees. "
    "The app's small sentiment-word rule can adjust the predicted class."
)
