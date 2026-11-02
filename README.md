# IMDb Movie Review Sentiment Analysis

## Project Overview

The IMDb Movie Review Sentiment Analysis project is a machine learning application that analyzes movie reviews and predicts whether the sentiment expressed in a review is **Positive** or **Negative**.

The project uses Natural Language Processing (NLP) techniques to convert text into numerical features and a machine learning model to classify sentiment. A Streamlit web application provides a simple interface where users can enter a movie review and view the prediction.

## Objectives

- Develop a machine learning model to classify movie reviews as positive or negative.
- Apply Natural Language Processing techniques to text data.
- Convert text into numerical features using TF-IDF vectorization.
- Evaluate the model using standard classification metrics.
- Develop an interactive web application using Streamlit.
- Understand how words and phrases influence sentiment predictions.

## Features

- **Sentiment Prediction:** Classifies a movie review as Positive or Negative.
- **Interactive Web Interface:** Allows users to enter reviews through a Streamlit application.
- **Text Processing:** Uses TF-IDF to convert review text into numerical features.
- **Model Evaluation:** Evaluates classification performance using standard metrics.
- **Dataset Analysis:** Includes a script for analyzing the dataset.
- **Model Evaluation Script:** Provides a separate script for evaluating the trained model.
- **Automated Tests:** Includes tests for application and model-related functionality.
- **Feature Analysis:** Allows inspection of learned model coefficients to identify words associated with positive and negative sentiment.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Scikit-learn | Machine learning and model evaluation |
| TF-IDF Vectorizer | Converts text into numerical features |
| Logistic Regression | Classifies movie review sentiment |
| Streamlit | Creates the web application |
| NumPy | Numerical operations |
| Pickle | Saves and loads the trained model and vectorizer |
| Pytest | Automated testing |

## Project Structure

```text
Sentiment Analysis/
│
├── aclImdb/
│   ├── train/
│   ├── test/
│   ├── imdb.vocab
│   └── imdbEr.txt
│
├── tests/
│   ├── test_app.py
│   └── test_train_model.py
│
├── app.py
├── analyze_dataset.py
├── evaluate_model.py
├── train_model.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: Virtual environment folders and `__pycache__` are local development files and generally do not need to be included in a GitHub repository.*

## Dataset

The project uses the IMDb movie review dataset, which contains movie reviews labeled as positive or negative.

The dataset is organized into training and testing directories.

- **Training Dataset:** Used to train the machine learning model.
- **Testing Dataset:** Used to evaluate the model on reviews that were not used for training.
- **Positive Reviews:** Reviews expressing favorable opinions about movies.
- **Negative Reviews:** Reviews expressing unfavorable opinions about movies.

The dataset is stored in the `aclImdb` directory.

## Methodology

The project follows these steps:

### 1. Data Collection

The IMDb movie review dataset is organized into positive and negative review folders for training and testing.

### 2. Text Preprocessing and Feature Extraction

The review text is converted into numerical features using the TF-IDF vectorization technique.

TF-IDF assigns numerical weights to words and phrases based on their importance within the text and the document collection.

### 3. Model Training

A Logistic Regression classifier learns patterns from the numerical representations of the labeled movie reviews.

### 4. Model Saving

The trained model and TF-IDF vectorizer are saved as:

- `model.pkl`
- `vectorizer.pkl`

These files allow the application to load the trained components without retraining the model every time it starts.

### 5. Sentiment Prediction

When a user enters a movie review, the application transforms the text using the saved vectorizer and passes the resulting features to the trained model.

The model predicts whether the review expresses positive or negative sentiment.

### 6. Model Evaluation

The model is evaluated using metrics such as accuracy, precision, recall, F1-score, balanced accuracy, and ROC-AUC.

## Model Performance

The following results were reported during model evaluation.

### Full Test Dataset

| Metric | Score |
|---|---:|
| Negative Class F1-Score | 90.44% |
| Positive Class F1-Score | 90.49% |
| Accuracy | 90.46% |
| Balanced Accuracy | 90.46% |
| ROC-AUC | 0.9659 |
| Number of Test Reviews | 25,000 |

### Deduplicated Test Dataset

| Metric | Score |
|---|---:|
| Negative Class F1-Score | 90.40% |
| Positive Class F1-Score | 90.55% |
| Accuracy | 90.47% |
| Balanced Accuracy | 90.47% |
| ROC-AUC | 0.9659 |
| Number of Test Reviews | 24,678 |

### Dataset Duplicate Analysis

The dataset analysis reported the following duplicate-review counts:

- **123 reviews** in the test dataset also appeared in the training dataset.
- **199 repeated copies** were identified within the test dataset.

The deduplicated evaluation provides an additional view of model performance after removing the identified duplicate test entries.

*These results reflect the reported evaluation output and should be updated if the model is retrained or the evaluation procedure changes.*

## Important Words Identified by the Model

Logistic Regression learns coefficients for the features used during training. These coefficients indicate how individual features influence the model's prediction.

For a binary Logistic Regression classifier, positive and negative coefficient directions depend on the order of the model's classes.

The following are previously reported example coefficients.

### Features Associated with Positive Sentiment

| Word or Phrase | Coefficient |
|---|---:|
| great | 3.699 |
| excellent | 2.776 |
| the best | 2.765 |
| best | 2.683 |
| fun | 2.395 |
| perfect | 2.311 |
| amazing | 2.080 |
| wonderful | 2.049 |
| well | 1.879 |
| loved | 1.789 |

### Features Associated with Negative Sentiment

| Word or Phrase | Coefficient |
|---|---:|
| bad | -4.782 |
| worst | -4.562 |
| the worst | -3.781 |
| awful | -3.491 |
| boring | -3.149 |
| waste | -3.062 |
| poor | -2.858 |
| no | -2.804 |
| nothing | -2.608 |
| dull | -2.301 |

*Note: These values are previously reported examples. Run the coefficient extraction script against the current saved model and vectorizer to verify the actual values before presenting them as final results.*

## Installation and Setup

### Prerequisites

Make sure Python is installed on your computer.

Check the installed version by running:

```bash
python --version
```

### 1. Open the Project Folder

Open the project folder in Visual Studio Code.

### 2. Create a Virtual Environment

Run:

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

On Windows PowerShell, run:

```powershell
.\.venv\Scripts\Activate.ps1
```

If the environment is already activated, you do not need to activate it again.

### 4. Install Dependencies

Run:

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Application

Run:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to use the application.

## Retraining the Model

To train the model again using the training dataset, run:

```bash
python train_model.py
```

After successful training, the script should save the trained model and vectorizer. The exact output depends on the implementation in `train_model.py`.

## Dataset Analysis

To analyze the dataset, run:

```bash
python analyze_dataset.py
```

This script can be used to inspect dataset information and identify potential data-quality issues, depending on its implementation.

## Model Evaluation

To evaluate the trained model, run:

```bash
python evaluate_model.py
```

The evaluation results depend on the metrics implemented in the script and the saved model being evaluated.

## Automated Testing

The project contains a `tests` directory for automated tests.

To run the tests using pytest, execute:

```bash
python -m pytest
```

Ensure that the required dependencies are installed before running the tests.

## Example

**Input Review:**

```text
The movie was amazing. The acting was excellent, and I really enjoyed the story.
```

**Expected Prediction:**

```text
Positive
```

The actual prediction depends on the trained model and its learned features.

## Learning Outcomes

Through this project, the following concepts can be understood:

- Fundamentals of Natural Language Processing.
- Text classification using machine learning.
- TF-IDF feature extraction.
- Logistic Regression for sentiment classification.
- Model training, saving, loading, and evaluation.
- Performance measurement using classification metrics.
- Building interactive machine learning applications with Streamlit.
- Understanding learned model coefficients and their relationship to sentiment predictions.