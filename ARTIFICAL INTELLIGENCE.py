"""
AI Sentiment Analysis of Online Reviews (fixed dataset loading)
===============================================================
Only download_dataset() and its helpers were changed. The experiment is the same.

Dataset sources tried in this order:
    1. Existing, complete  aclImdb  folder
    2. Local  aclImdb_v1.tar.gz
    3. Local  train-00000-of-00001.parquet + test-00000-of-00001.parquet
    4. Stanford (short timeout)
    5. Hugging Face, then hf-mirror.com
"""

import os
import re
import json
import time
import random
import shutil
import tarfile
from io import BytesIO

import requests
import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

from statsmodels.stats.contingency_tables import mcnemar


# ============================================================
# GLOBAL SETTINGS
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATASET_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
ARCHIVE_FILE = "aclImdb_v1.tar.gz"
DATASET_FOLDER = "aclImdb"
DATASET_TEMP_FOLDER = "aclImdb_building"

PARQUET_FILES = {
    "train": "train-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}
HF_BASES = [
    "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/",
    "https://hf-mirror.com/datasets/stanfordnlp/imdb/resolve/main/plain_text/",
]

RESULTS_FOLDER = "results"
MODELS_FOLDER = os.path.join(RESULTS_FOLDER, "models")
PREDICTIONS_FOLDER = os.path.join(RESULTS_FOLDER, "predictions")
CACHE_FOLDER = os.path.join(RESULTS_FOLDER, "pipeline_cache")
DISTILBERT_FILE = os.path.join(RESULTS_FOLDER, "distilbert_predictions.csv")

for folder in [RESULTS_FOLDER, MODELS_FOLDER, PREDICTIONS_FOLDER, CACHE_FOLDER]:
    os.makedirs(folder, exist_ok=True)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

TFIDF_SETTINGS = {
    "max_features": 50000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
    "dtype": np.float32
}

CV_FOLDS = 5
RF_CV_FOLDS = 3
RF_RANDOM_SEARCH_ITERATIONS = 6

RUN_PREPROCESSING_EXPERIMENT = True
RUN_RF_SEED_EXPERIMENT = True
RF_SEEDS = [21, 42, 84]

NEGATION_WINDOW = 3
CV_NEAR_TIE_TOLERANCE = 0.001


# ============================================================
# HELPER
# ============================================================

def safe_filename(name):
    return name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


# ============================================================
# DATASET: EXTRACTION, DOWNLOAD, BUILD
# ============================================================

def safe_extract_tar(archive_path, destination):
    destination = os.path.abspath(destination)

    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            member_path = os.path.abspath(os.path.join(destination, member.name))
            if os.path.commonpath([destination, member_path]) != destination:
                raise ValueError("Unsafe path detected in archive.")

        try:
            archive.extractall(destination, filter="data")   # Python 3.12+
        except TypeError:
            archive.extractall(destination)                  # older Python


def dataset_is_complete(folder=DATASET_FOLDER):
    """True only if all four review folders exist and contain .txt files."""
    for split in ["train", "test"]:
        for label in ["pos", "neg"]:
            path = os.path.join(folder, split, label)
            if not os.path.isdir(path):
                return False
            if not any(name.endswith(".txt") for name in os.listdir(path)):
                return False
    return True


def build_from_parquet(train_bytes, test_bytes):
    """Create the aclImdb folder structure from the two parquet files."""
    if os.path.exists(DATASET_TEMP_FOLDER):
        shutil.rmtree(DATASET_TEMP_FOLDER, ignore_errors=True)

    for split, data in [("train", train_bytes), ("test", test_bytes)]:
        df = pd.read_parquet(BytesIO(data))

        for label_name in ["neg", "pos"]:
            os.makedirs(os.path.join(DATASET_TEMP_FOLDER, split, label_name), exist_ok=True)

        for i, row in enumerate(df.itertuples(index=False)):
            label_name = "pos" if row.label == 1 else "neg"
            path = os.path.join(DATASET_TEMP_FOLDER, split, label_name, f"{i}_{row.label}.txt")
            with open(path, "w", encoding="utf-8") as file:
                file.write(row.text)

        print(f"  {split}: {len(df)} reviews written.")

    # Rename only after everything was written successfully.
    if os.path.exists(DATASET_FOLDER):
        shutil.rmtree(DATASET_FOLDER, ignore_errors=True)
    os.replace(DATASET_TEMP_FOLDER, DATASET_FOLDER)


def try_download(url, timeout=(15, 300), attempts=2):
    """Download a URL into memory. Returns bytes, or None if it failed."""
    for attempt in range(1, attempts + 1):
        try:
            print(f"Trying ({attempt}/{attempts}): {url}")
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as error:
            print(f"  failed: {error.__class__.__name__}")
            time.sleep(2)
    return None


def download_dataset():

    # 1. Already extracted and complete
    if dataset_is_complete():
        print("IMDb dataset already exists. Skipping download.")
        return

    if os.path.exists(DATASET_FOLDER):
        print("Found an incomplete aclImdb folder. Rebuilding it.")
        shutil.rmtree(DATASET_FOLDER, ignore_errors=True)

    # 2. Manually downloaded tar.gz
    if os.path.exists(ARCHIVE_FILE):
        print("Using local archive. Extracting...")
        try:
            safe_extract_tar(ARCHIVE_FILE, ".")
            if dataset_is_complete():
                print("Dataset extracted.")
                return
        except (tarfile.TarError, OSError, EOFError) as error:
            print(f"  local archive could not be used: {error}")
        shutil.rmtree(DATASET_FOLDER, ignore_errors=True)

    # 3. Manually downloaded parquet files
    if all(os.path.exists(name) for name in PARQUET_FILES.values()):
        print("Using local parquet files...")
        with open(PARQUET_FILES["train"], "rb") as f:
            train_bytes = f.read()
        with open(PARQUET_FILES["test"], "rb") as f:
            test_bytes = f.read()
        build_from_parquet(train_bytes, test_bytes)
        print("Dataset built.")
        return

    # 4. Stanford (short timeout so it fails fast when blocked)
    data = try_download(DATASET_URL, timeout=(10, 600), attempts=1)
    if data is not None:
        with open(ARCHIVE_FILE, "wb") as f:
            f.write(data)
        safe_extract_tar(ARCHIVE_FILE, ".")
        print("Dataset extracted.")
        return

    # 5. Hugging Face, then mirror
    for base in HF_BASES:
        train_bytes = try_download(base + PARQUET_FILES["train"])
        test_bytes = try_download(base + PARQUET_FILES["test"]) if train_bytes else None
        if train_bytes and test_bytes:
            print("Building dataset folders...")
            build_from_parquet(train_bytes, test_bytes)
            print("Dataset built.")
            return

    raise SystemExit(
        "\n" + "=" * 70 + "\n"
        "COULD NOT GET THE DATASET FROM ANY SOURCE.\n"
        "Your network is blocking all of them. Do ONE of these:\n"
        "  A) Connect to a phone hotspot (or other Wi-Fi) and run again.\n"
        "  B) On another device, download these two files:\n"
        "       https://huggingface.co/datasets/stanfordnlp/imdb/tree/main/plain_text\n"
        "       train-00000-of-00001.parquet\n"
        "       test-00000-of-00001.parquet\n"
        "     Copy both into the folder next to this script, then run again.\n"
        "  C) Put aclImdb_v1.tar.gz next to this script, then run again.\n"
        + "=" * 70
    )


# ============================================================
# LOAD REVIEWS
# ============================================================

def load_reviews(folder, label):
    reviews = []
    labels = []

    files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])

    for index, filename in enumerate(files, start=1):
        file_path = os.path.join(folder, filename)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
                reviews.append(file.read())
            labels.append(label)
        except OSError as error:
            print(f"Could not read {file_path}: {error}")

        if index % 5000 == 0:
            print(f"Loaded {index} files from {os.path.basename(folder)}...")

    return reviews, labels


def load_dataset():
    print()
    print("Loading training data...")

    pos_r, pos_l = load_reviews(os.path.join(DATASET_FOLDER, "train", "pos"), 1)
    neg_r, neg_l = load_reviews(os.path.join(DATASET_FOLDER, "train", "neg"), 0)
    train_reviews = pos_r + neg_r
    train_labels = pos_l + neg_l
    print(f"Training reviews: {len(train_reviews)}")

    print()
    print("Loading testing data...")

    pos_r, pos_l = load_reviews(os.path.join(DATASET_FOLDER, "test", "pos"), 1)
    neg_r, neg_l = load_reviews(os.path.join(DATASET_FOLDER, "test", "neg"), 0)
    test_reviews = pos_r + neg_r
    test_labels = pos_l + neg_l
    print(f"Testing reviews: {len(test_reviews)}")

    return train_reviews, train_labels, test_reviews, test_labels


def create_dataframes(train_reviews, train_labels, test_reviews, test_labels):
    train_df = pd.DataFrame({"review": train_reviews, "sentiment": train_labels})
    test_df = pd.DataFrame({"review": test_reviews, "sentiment": test_labels})

    # Reproducible training shuffle.
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    return train_df, test_df


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text, style="standard"):
    text = text.lower()
    text = re.sub(r"<br\s*/?>", " ", text)

    if style == "standard":
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

    elif style == "negation":
        text = re.sub(r"[^a-zA-Z'\s]", " ", text)

        replacements = {
            "isn't": "is not", "wasn't": "was not", "aren't": "are not",
            "weren't": "were not", "don't": "do not", "doesn't": "does not",
            "didn't": "did not", "can't": "can not", "couldn't": "could not",
            "wouldn't": "would not", "shouldn't": "should not", "won't": "will not",
            "haven't": "have not", "hasn't": "has not", "hadn't": "had not"
        }
        for old, new in replacements.items():
            text = text.replace(old, new)

        negation_words = {"not", "never", "no"}
        processed_words = []
        negate_remaining = 0

        for word in text.split():
            if word in negation_words:
                processed_words.append(word)
                negate_remaining = NEGATION_WINDOW
            elif negate_remaining > 0:
                processed_words.append("NEG_" + word)
                negate_remaining -= 1
            else:
                processed_words.append(word)

        text = " ".join(processed_words)

    else:
        raise ValueError("style must be 'standard' or 'negation'")

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_text_data(train_df, test_df, style="standard"):
    print()
    print(f"Preprocessing style: {style}")

    train_copy = train_df.copy()
    test_copy = test_df.copy()

    train_copy["clean_review"] = train_copy["review"].apply(lambda t: clean_text(t, style=style))
    test_copy["clean_review"] = test_copy["review"].apply(lambda t: clean_text(t, style=style))

    print("Text preprocessing completed.")

    return (
        train_copy, test_copy,
        train_copy["clean_review"], train_copy["sentiment"],
        test_copy["clean_review"], test_copy["sentiment"]
    )


# ============================================================
# CV, TF-IDF, PIPELINE
# ============================================================

def create_cv(folds=CV_FOLDS):
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)


def create_vectorizer():
    return TfidfVectorizer(**TFIDF_SETTINGS)


def make_pipeline(classifier):
    return Pipeline(
        steps=[("tfidf", create_vectorizer()), ("classifier", classifier)],
        memory=CACHE_FOLDER
    )


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def create_models():
    return {
        "Logistic Regression": {
            "pipeline": make_pipeline(LogisticRegression(max_iter=2000, random_state=SEED)),
            "params": {"classifier__C": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]},
            "search": "grid",
            "folds": 5
        },
        "Multinomial Naive Bayes": {
            "pipeline": make_pipeline(MultinomialNB()),
            "params": {"classifier__alpha": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]},
            "search": "grid",
            "folds": 5
        },
        "Linear SVM": {
            "pipeline": make_pipeline(LinearSVC(random_state=SEED)),
            "params": {"classifier__C": [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]},
            "search": "grid",
            "folds": 5
        },
        "Random Forest": {
            "pipeline": make_pipeline(RandomForestClassifier(random_state=SEED, n_jobs=1)),
            "params": {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [None, 30, 60],
                "classifier__min_samples_leaf": [1, 2, 4]
            },
            "search": "random",
            "folds": RF_CV_FOLDS,
            "n_iter": RF_RANDOM_SEARCH_ITERATIONS
        }
    }


# ============================================================
# GENERIC MODEL TUNING
# ============================================================

def tune_model(name, specification, X_train, y_train):
    print()
    print("=" * 70)
    print(f"TUNING: {name}")
    print("=" * 70)

    pipeline = specification["pipeline"]
    params = specification["params"]
    cv = create_cv(specification["folds"])

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    if specification["search"] == "random":
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=params,
            n_iter=specification["n_iter"],
            scoring=scoring,
            refit="f1",
            cv=cv,
            random_state=SEED,
            n_jobs=1,
            verbose=2,
            return_train_score=False,
            error_score="raise"
        )
    else:
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=1,
            verbose=2,
            return_train_score=False,
            error_score="raise"
        )

    start_time = time.time()
    search.fit(X_train, y_train)
    tuning_time = time.time() - start_time

    cv_table = pd.DataFrame(search.cv_results_).sort_values("rank_test_f1")
    cv_table.to_csv(
        os.path.join(RESULTS_FOLDER, safe_filename(name) + "_cv_results.csv"),
        index=False
    )

    best_index = search.best_index_

    summary = {
        "Model": name,
        "CV Mean F1": search.cv_results_["mean_test_f1"][best_index],
        "CV Std F1": search.cv_results_["std_test_f1"][best_index],
        "CV Mean Accuracy": search.cv_results_["mean_test_accuracy"][best_index],
        "CV Mean ROC AUC": search.cv_results_["mean_test_roc_auc"][best_index],
        "CV Search + Best Refit Time (seconds)": tuning_time,
        "Best Estimator Refit Time (seconds)": search.refit_time_,
        "Best Parameters": json.dumps(search.best_params_, default=str)
    }

    print()
    print(f"Best mean CV F1: {search.best_score_:.4f}")
    print(f"CV F1 standard deviation: {summary['CV Std F1']:.4f}")
    print(f"Tuning time: {tuning_time:.2f} seconds")
    print(f"Final refit time: {search.refit_time_:.2f} seconds")
    print("Best parameters:")
    for key, value in search.best_params_.items():
        print(f"  {key}: {value}")

    return search.best_estimator_, summary, search


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, predictions, scores=None):
    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)

    auc = np.nan
    if scores is not None:
        scores = np.asarray(scores)
        if len(np.unique(scores)) < 2:
            auc = 0.5
        else:
            try:
                auc = roc_auc_score(y_true, scores)
            except ValueError:
                auc = np.nan

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    }


def get_score_values(model, X_test, predictions):
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        classes = list(model.classes_)
        if probabilities.ndim == 2 and 1 in classes:
            return probabilities[:, classes.index(1)]

    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X_test))

    return np.asarray(predictions, dtype=float)


def create_result_row(name, metrics, cv_mean, cv_std, tuning_time, fit_time, test_time, parameters):
    return {
        "Model": name,
        "CV Mean F1": cv_mean,
        "CV Std F1": cv_std,
        "Test Accuracy": metrics["Accuracy"],
        "Test Precision": metrics["Precision"],
        "Test Recall": metrics["Recall"],
        "Test F1": metrics["F1 Score"],
        "Test ROC AUC": metrics["ROC AUC"],
        "CV Search + Best Refit Time (seconds)": tuning_time,
        "Best Estimator Refit Time (seconds)": fit_time,
        "Pipeline Test Predict + Score Time (seconds)": test_time,
        "Best Parameters": parameters
    }


def save_predictions(model_name, y_true, predictions, scores):
    output = pd.DataFrame({
        "test_index": np.arange(len(y_true)),
        "actual": np.asarray(y_true),
        "predicted": np.asarray(predictions),
        "score": np.asarray(scores)
    })
    output.to_csv(
        os.path.join(PREDICTIONS_FOLDER, safe_filename(model_name) + "_predictions.csv"),
        index=False
    )


def save_model(model_name, model):
    path = os.path.join(MODELS_FOLDER, safe_filename(model_name) + ".joblib")
    joblib.dump(model, path)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

def final_model_evaluation(name, model, cv_summary, X_test, y_test):
    """The search already refitted the winner on the full training set
    (refit='f1'), so this only evaluates on the official test set."""

    print()
    print("=" * 70)
    print(f"FINAL TEST: {name}")
    print("=" * 70)

    start_test = time.time()
    predictions = model.predict(X_test)
    scores = get_score_values(model, X_test, predictions)
    test_time = time.time() - start_test

    metrics = calculate_metrics(y_test, predictions, scores)

    print()
    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1 Score : {metrics['F1 Score']:.4f}")
    print(f"ROC AUC  : {metrics['ROC AUC']:.4f}")
    print()
    print(classification_report(
        y_test, predictions, labels=[0, 1],
        target_names=["Negative", "Positive"], zero_division=0
    ))

    row = create_result_row(
        name=name,
        metrics=metrics,
        cv_mean=cv_summary["CV Mean F1"],
        cv_std=cv_summary["CV Std F1"],
        tuning_time=cv_summary["CV Search + Best Refit Time (seconds)"],
        fit_time=cv_summary["Best Estimator Refit Time (seconds)"],
        test_time=test_time,
        parameters=cv_summary["Best Parameters"]
    )

    save_predictions(name, y_test, predictions, scores)
    save_model(name, model)

    return row, predictions, scores, model


# ============================================================
# DUMMY BASELINES
# ============================================================

def evaluate_dummy_baselines(y_train, y_test):
    models = {
        "Dummy Most Frequent": DummyClassifier(strategy="most_frequent", random_state=SEED),
        "Dummy Stratified": DummyClassifier(strategy="stratified", random_state=SEED)
    }

    X_dummy_train = np.zeros((len(y_train), 1), dtype=np.float32)
    X_dummy_test = np.zeros((len(y_test), 1), dtype=np.float32)

    outputs = []

    for name, model in models.items():
        print()
        print(f"Evaluating {name}...")

        start_fit = time.time()
        model.fit(X_dummy_train, y_train)
        fit_time = time.time() - start_fit

        start_test = time.time()
        predictions = model.predict(X_dummy_test)

        try:
            probabilities = model.predict_proba(X_dummy_test)
            classes = list(model.classes_)
            if 1 in classes:
                scores = probabilities[:, classes.index(1)]
            else:
                scores = predictions.astype(float)
        except (AttributeError, IndexError):
            scores = predictions.astype(float)

        test_time = time.time() - start_test

        metrics = calculate_metrics(y_test, predictions, scores)

        row = create_result_row(
            name=name, metrics=metrics, cv_mean=np.nan, cv_std=np.nan,
            tuning_time=np.nan, fit_time=np.nan, test_time=np.nan,
            parameters="Baseline"
        )
        row["Dummy Fit Time (seconds)"] = fit_time
        row["Dummy Predict + Score Time (seconds)"] = test_time

        save_predictions(name, y_test, predictions, scores)
        outputs.append((row, name, predictions, scores))

    return outputs


# ============================================================
# VADER
# ============================================================

def score_vader_reviews(reviews, analyzer):
    return np.array(
        [analyzer.polarity_scores(review)["compound"] for review in reviews],
        dtype=np.float32
    )


def choose_vader_threshold(train_scores, y_train):
    """Choose the VADER decision threshold using TRAINING DATA ONLY."""
    thresholds = np.round(np.arange(-0.50, 0.501, 0.01), 2)
    rows = []

    for threshold in thresholds:
        predictions = (train_scores >= threshold).astype(int)
        rows.append({
            "Threshold": float(threshold),
            "Training Accuracy": accuracy_score(y_train, predictions),
            "Training F1": f1_score(y_train, predictions, zero_division=0)
        })

    table = pd.DataFrame(rows).sort_values(
        ["Training F1", "Training Accuracy"], ascending=[False, False]
    ).reset_index(drop=True)
    table.to_csv(os.path.join(RESULTS_FOLDER, "vader_training_threshold_search.csv"), index=False)
    return float(table.iloc[0]["Threshold"]), table


def evaluate_vader(train_reviews, y_train, test_reviews, y_test):
    print()
    print("=" * 70)
    print("EVALUATING VADER")
    print("=" * 70)

    analyzer = SentimentIntensityAnalyzer()

    train_start = time.time()
    train_scores = score_vader_reviews(train_reviews, analyzer)
    train_scoring_time = time.time() - train_start

    threshold_start = time.time()
    tuned_threshold, threshold_table = choose_vader_threshold(train_scores, y_train)
    threshold_selection_time = time.time() - threshold_start

    test_start = time.time()
    test_scores = score_vader_reviews(test_reviews, analyzer)
    test_scoring_time = time.time() - test_start

    outputs = []
    for name, threshold in [("VADER Default", 0.0), ("VADER Tuned", tuned_threshold)]:
        test_predictions = (test_scores >= threshold).astype(int)
        metrics = calculate_metrics(y_test, test_predictions, test_scores)

        row = create_result_row(
            name=name, metrics=metrics, cv_mean=np.nan, cv_std=np.nan,
            tuning_time=np.nan, fit_time=np.nan, test_time=np.nan,
            parameters=f"Rule-based VADER; positive if compound >= {threshold:.2f}"
        )
        row["VADER Training Scoring Time (seconds)"] = train_scoring_time
        row["VADER Threshold Selection Time (seconds)"] = threshold_selection_time
        row["VADER Test Scoring Time (seconds)"] = test_scoring_time
        row["VADER Threshold"] = threshold

        save_predictions(name, y_test, test_predictions, test_scores)
        outputs.append((row, name, test_predictions, test_scores))

        print(f"{name}: threshold={threshold:.2f}, F1={metrics['F1 Score']:.4f}, "
              f"accuracy={metrics['Accuracy']:.4f}")

    print(f"Training-only tuned VADER threshold: {tuned_threshold:.2f}")
    return outputs


# ============================================================
# PLOTS
# ============================================================

def save_confusion_matrix(name, y_true, predictions):
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(
        os.path.join(RESULTS_FOLDER, safe_filename(name) + "_confusion_matrix.png"),
        dpi=300, bbox_inches="tight"
    )
    plt.close()


def create_model_comparison_graph(results):
    plot_data = results[
        ["Model", "Test Accuracy", "Test Precision", "Test Recall", "Test F1"]
    ].melt(id_vars="Model", var_name="Metric", value_name="Score")

    plt.figure(figsize=(14, 7))
    sns.barplot(data=plot_data, x="Model", y="Score", hue="Metric")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.title("Final Sentiment Model Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, "model_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()


# ============================================================
# ERROR ANALYSIS
# ============================================================

def perform_error_analysis(test_df, y_test, predictions, model_name, sample_size=50):
    actual = np.asarray(y_test)
    predictions = np.asarray(predictions)

    fp_count = int(((actual == 0) & (predictions == 1)).sum())
    fn_count = int(((actual == 1) & (predictions == 0)).sum())

    print()
    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    print(f"False positives: {fp_count}")
    print(f"False negatives: {fn_count}")

    error_mask = actual != predictions

    errors = test_df.loc[error_mask, ["review", "sentiment"]].copy()
    errors["predicted"] = predictions[error_mask]
    errors["error_type"] = np.where(errors["sentiment"] == 0, "False Positive", "False Negative")

    # Label manually later: negation / sarcasm / mixed sentiment / plot summary / other
    errors["category"] = ""

    sampled_errors = errors.sample(n=min(sample_size, len(errors)), random_state=SEED)
    sampled_errors.to_csv(
        os.path.join(RESULTS_FOLDER, safe_filename(model_name) + "_sampled_errors.csv"),
        index=False
    )

    pd.DataFrame([{
        "Model": model_name,
        "False Positives": fp_count,
        "False Negatives": fn_count,
        "Total Errors": fp_count + fn_count
    }]).to_csv(
        os.path.join(RESULTS_FOLDER, safe_filename(model_name) + "_error_counts.csv"),
        index=False
    )


# ============================================================
# NEGATION PREPROCESSING EXPERIMENT
# ============================================================

def run_preprocessing_experiment(train_df, test_df):
    """Re-tune C independently for standard and negation text, for LR and SVM."""
    print()
    print("=" * 70)
    print("PREPROCESSING / NEGATION EXPERIMENT")
    print("=" * 70)

    model_specs = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=SEED),
            [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
        ),
        "Linear SVM": (
            LinearSVC(random_state=SEED),
            [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        )
    }

    rows = []
    prediction_outputs = {}

    for style in ["standard", "negation"]:
        _, _, X_train_style, y_train_style, X_test_style, y_test_style = prepare_text_data(
            train_df, test_df, style=style
        )

        for model_name, (classifier, c_grid) in model_specs.items():
            experiment_name = f"{model_name} - {style}"
            print(f"Tuning {experiment_name}...")

            search = GridSearchCV(
                make_pipeline(classifier),
                {"classifier__C": c_grid},
                scoring="f1",
                refit=True,
                cv=create_cv(5),
                n_jobs=1,
                verbose=0,
                error_score="raise"
            )

            search_start = time.time()
            search.fit(X_train_style, y_train_style)
            search_time = time.time() - search_start

            predict_start = time.time()
            preds = search.best_estimator_.predict(X_test_style)
            predict_time = time.time() - predict_start

            rows.append({
                "Model": model_name,
                "Preprocessing": style,
                "Best C": search.best_params_["classifier__C"],
                "CV Mean F1": search.best_score_,
                "CV Std F1": search.cv_results_["std_test_score"][search.best_index_],
                "Test F1": f1_score(y_test_style, preds, zero_division=0),
                "CV Search + Best Refit Time (seconds)": search_time,
                "Best Estimator Refit Time (seconds)": search.refit_time_,
                "Pipeline Test Predict Time (seconds)": predict_time
            })
            prediction_outputs[experiment_name] = preds

            save_predictions(
                experiment_name, y_test_style, preds,
                get_score_values(search.best_estimator_, X_test_style, preds)
            )

    output = pd.DataFrame(rows)
    output.to_csv(os.path.join(RESULTS_FOLDER, "preprocessing_experiment.csv"), index=False)
    print(output.to_string(index=False))
    return output, prediction_outputs


# ============================================================
# RANDOM FOREST SEED STABILITY
# ============================================================

def random_forest_seed_experiment(X_train, y_train, X_test, y_test, best_params):
    print()
    print("=" * 70)
    print("RANDOM FOREST SEED STABILITY")
    print("=" * 70)

    rows = []

    for seed in RF_SEEDS:
        print()
        print(f"Random Forest seed: {seed}")

        classifier = RandomForestClassifier(
            n_estimators=best_params.get("classifier__n_estimators", 200),
            max_depth=best_params.get("classifier__max_depth", None),
            min_samples_leaf=best_params.get("classifier__min_samples_leaf", 1),
            random_state=seed,
            n_jobs=1
        )

        model = make_pipeline(classifier)

        fit_start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - fit_start

        test_start = time.time()
        predictions = model.predict(X_test)
        test_time = time.time() - test_start

        rows.append({
            "Seed": seed,
            "Accuracy": accuracy_score(y_test, predictions),
            "F1": f1_score(y_test, predictions),
            "Fit Time (seconds)": fit_time,
            "Pipeline Test Predict + Score Time (seconds)": test_time
        })

    output = pd.DataFrame(rows)
    output.to_csv(os.path.join(RESULTS_FOLDER, "random_forest_seed_stability.csv"), index=False)
    print()
    print(output.to_string(index=False))


# ============================================================
# McNEMAR SIGNIFICANCE TEST
# ============================================================

def run_mcnemar_test(y_true, predictions_a, predictions_b, name_a, name_b):
    actual = np.asarray(y_true)
    a_correct = np.asarray(predictions_a) == actual
    b_correct = np.asarray(predictions_b) == actual

    both_correct = int(np.sum(a_correct & b_correct))
    a_correct_b_wrong = int(np.sum(a_correct & ~b_correct))
    a_wrong_b_correct = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))

    table = [[both_correct, a_correct_b_wrong],
             [a_wrong_b_correct, both_wrong]]

    result = mcnemar(table, exact=False, correction=True)

    return {
        "Model A": name_a,
        "Model B": name_b,
        "Both Correct": both_correct,
        "A Correct B Wrong": a_correct_b_wrong,
        "A Wrong B Correct": a_wrong_b_correct,
        "Both Wrong": both_wrong,
        "Statistic": result.statistic,
        "P Value": result.pvalue,
        "Significant at 0.05": result.pvalue < 0.05
    }


# ============================================================
# OPTIONAL DISTILBERT IMPORT
# ============================================================

def load_optional_distilbert(y_test):
    if not os.path.exists(DISTILBERT_FILE):
        print()
        print("No DistilBERT prediction file found.")
        print("Skipping optional DistilBERT comparison.")
        return None

    data = pd.read_csv(DISTILBERT_FILE)

    if not {"test_index", "actual", "predicted", "score"}.issubset(data.columns):
        print("DistilBERT CSV is missing required columns.")
        return None

    if len(data) != len(y_test):
        print("DistilBERT prediction count does not match the test set.")
        return None

    if not np.array_equal(data["actual"].to_numpy(), np.asarray(y_test)):
        print("DistilBERT labels do not match the current test-set order.")
        return None

    metrics = calculate_metrics(y_test, data["predicted"], data["score"])

    row = create_result_row(
        name="DistilBERT", metrics=metrics, cv_mean=np.nan, cv_std=np.nan,
        tuning_time=np.nan, fit_time=np.nan, test_time=np.nan,
        parameters="External Colab experiment"
    )

    return row, data["predicted"].to_numpy(), data["score"].to_numpy()


# ============================================================
# TEST NEW REVIEWS
# ============================================================

def test_new_reviews(model, style="standard"):
    reviews = [
        "This movie was absolutely fantastic and I loved every minute of it.",
        "The film was terrible, boring and a complete waste of time.",
        "The acting was good but the story was not very interesting.",
        "An excellent movie with brilliant performances and a great story."
    ]

    cleaned = [clean_text(review, style=style) for review in reviews]
    predictions = model.predict(cleaned)

    print()
    print("=" * 70)
    print("NEW REVIEW PREDICTIONS")
    print("=" * 70)

    for review, prediction in zip(reviews, predictions):
        sentiment = "Positive" if prediction == 1 else "Negative"
        print()
        print(f"Review: {review}")
        print(f"Prediction: {sentiment}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AI SENTIMENT ANALYSIS EXPERIMENT")
    print("=" * 70)

    # Start with an empty pipeline cache so old cached transforms are never reused.
    if os.path.exists(CACHE_FOLDER):
        shutil.rmtree(CACHE_FOLDER, ignore_errors=True)
    os.makedirs(CACHE_FOLDER, exist_ok=True)

    # LOAD
    download_dataset()

    train_reviews, train_labels, test_reviews, test_labels = load_dataset()

    train_df, test_df = create_dataframes(train_reviews, train_labels, test_reviews, test_labels)

    print()
    print("Training shape:", train_df.shape)
    print("Testing shape:", test_df.shape)
    print()
    print("Training class distribution:")
    print(train_df["sentiment"].value_counts())

    # PREPROCESS
    train_clean, test_clean, X_train, y_train, X_test, y_test = prepare_text_data(
        train_df, test_df, style="standard"
    )

    # TUNE CLASSICAL MODELS
    specifications = create_models()

    result_rows = []
    predictions = {}
    scores = {}
    fitted_models = {}
    tuning_objects = {}
    cv_summaries = {}

    for name, specification in specifications.items():
        selected_model, cv_summary, search_object = tune_model(name, specification, X_train, y_train)

        cv_summaries[name] = cv_summary
        tuning_objects[name] = search_object

        result_row, model_predictions, model_scores, fitted_model = final_model_evaluation(
            name, selected_model, cv_summary, X_test, y_test
        )

        result_rows.append(result_row)
        predictions[name] = model_predictions
        scores[name] = model_scores
        fitted_models[name] = fitted_model

        save_confusion_matrix(name, y_test, model_predictions)

    # BASELINES
    for row, name, model_predictions, model_scores in evaluate_dummy_baselines(y_train, y_test):
        result_rows.append(row)
        predictions[name] = model_predictions
        scores[name] = model_scores
        save_confusion_matrix(name, y_test, model_predictions)

    # VADER
    vader_outputs = evaluate_vader(train_df["review"], y_train, test_df["review"], y_test)

    for vader_row, vader_name, vader_predictions, vader_scores in vader_outputs:
        result_rows.append(vader_row)
        predictions[vader_name] = vader_predictions
        scores[vader_name] = vader_scores
        save_confusion_matrix(vader_name, y_test, vader_predictions)

    # OPTIONAL DISTILBERT
    distilbert_result = load_optional_distilbert(y_test)

    if distilbert_result is not None:
        distilbert_row, distilbert_predictions, distilbert_scores = distilbert_result
        result_rows.append(distilbert_row)
        predictions["DistilBERT"] = distilbert_predictions
        scores["DistilBERT"] = distilbert_scores

    # RESULTS TABLE
    results_table = (
        pd.DataFrame(result_rows)
        .sort_values("Test F1", ascending=False)
        .reset_index(drop=True)
    )
    results_table.to_csv(os.path.join(RESULTS_FOLDER, "results_table.csv"), index=False)

    print()
    print("=" * 70)
    print("FINAL RESULTS TABLE")
    print("=" * 70)
    print()
    print(results_table.to_string(index=False))

    create_model_comparison_graph(results_table)

    # SELECT BEST CLASSICAL MODEL USING CV F1
    classical_cv = pd.DataFrame([
        {"Model": name, "CV Mean F1": s["CV Mean F1"], "CV Std F1": s["CV Std F1"]}
        for name, s in cv_summaries.items()
    ]).sort_values("CV Mean F1", ascending=False).reset_index(drop=True)

    classical_cv.to_csv(os.path.join(RESULTS_FOLDER, "classical_model_cv_ranking.csv"), index=False)

    if len(classical_cv) >= 2:
        cv_gap = float(classical_cv.iloc[0]["CV Mean F1"] - classical_cv.iloc[1]["CV Mean F1"])
        near_tie = cv_gap <= CV_NEAR_TIE_TOLERANCE
        pd.DataFrame([{
            "Top Model": classical_cv.iloc[0]["Model"],
            "Second Model": classical_cv.iloc[1]["Model"],
            "CV F1 Gap": cv_gap,
            "Near Tie Tolerance": CV_NEAR_TIE_TOLERANCE,
            "Effectively Near-Tied": near_tie,
            "Interpretation": (
                "Leading models have practically similar CV F1; avoid claiming a "
                "meaningful advantage without paired significance evidence."
                if near_tie else
                "CV F1 gap exceeds the predeclared near-tie tolerance."
            )
        }]).to_csv(os.path.join(RESULTS_FOLDER, "classical_model_near_tie_check.csv"), index=False)

        print()
        print(f"Top-two CV F1 gap: {cv_gap:.6f}")
        if near_tie:
            print("Interpretation: effectively near-tied by the predeclared 0.001 CV-F1 tolerance.")

    best_classical_name = classical_cv.iloc[0]["Model"]
    best_classical_model = fitted_models[best_classical_name]

    print()
    print("=" * 70)
    print("BEST CLASSICAL MODEL BY CV F1")
    print("=" * 70)
    print(best_classical_name)
    print(f"CV F1 = {classical_cv.iloc[0]['CV Mean F1']:.4f}")

    # ERROR ANALYSIS
    perform_error_analysis(
        test_df, y_test, predictions[best_classical_name], best_classical_name, sample_size=50
    )

    # NEGATION EXPERIMENT
    preprocessing_predictions = {}

    if RUN_PREPROCESSING_EXPERIMENT:
        preprocessing_table, preprocessing_predictions = run_preprocessing_experiment(train_df, test_df)

    # RANDOM FOREST SEED STABILITY
    if RUN_RF_SEED_EXPERIMENT:
        random_forest_seed_experiment(
            X_train, y_train, X_test, y_test,
            tuning_objects["Random Forest"].best_params_
        )

    # SIGNIFICANCE TESTING
    significance_results = []

    significance_results.append(run_mcnemar_test(
        y_test, predictions["Logistic Regression"], predictions["Linear SVM"],
        "Logistic Regression", "Linear SVM"
    ))

    significance_results.append(run_mcnemar_test(
        y_test, predictions[best_classical_name], predictions["VADER Tuned"],
        best_classical_name, "VADER Tuned"
    ))

    if preprocessing_predictions:
        for model_name in ["Logistic Regression", "Linear SVM"]:
            standard_key = f"{model_name} - standard"
            negation_key = f"{model_name} - negation"
            if standard_key in preprocessing_predictions and negation_key in preprocessing_predictions:
                significance_results.append(run_mcnemar_test(
                    y_test,
                    preprocessing_predictions[standard_key],
                    preprocessing_predictions[negation_key],
                    standard_key, negation_key
                ))

    if "DistilBERT" in predictions:
        significance_results.append(run_mcnemar_test(
            y_test, predictions[best_classical_name], predictions["DistilBERT"],
            best_classical_name, "DistilBERT"
        ))

    if "DistilBERT" in predictions and preprocessing_predictions:
        for negation_key in ["Logistic Regression - negation", "Linear SVM - negation"]:
            if negation_key in preprocessing_predictions:
                significance_results.append(run_mcnemar_test(
                    y_test, preprocessing_predictions[negation_key],
                    predictions["DistilBERT"], negation_key, "DistilBERT"
                ))

    significance_df = pd.DataFrame(significance_results)
    significance_df.to_csv(os.path.join(RESULTS_FOLDER, "mcnemar_tests.csv"), index=False)

    print()
    print("=" * 70)
    print("McNEMAR SIGNIFICANCE TESTS")
    print("=" * 70)
    print()
    print(significance_df.to_string(index=False))

    # NEW REVIEWS
    test_new_reviews(best_classical_model, style="standard")

    # CACHE CLEANUP
    print()
    print("Cleaning temporary pipeline cache...")
    if os.path.exists(CACHE_FOLDER):
        shutil.rmtree(CACHE_FOLDER, ignore_errors=True)

    # COMPLETE
    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()
    print(f"Best classical model by CV F1: {best_classical_name}")
    print()
    print("Main results file:")
    print(os.path.join(RESULTS_FOLDER, "results_table.csv"))
    print()
    print("The results folder also contains:")
    for item in [
        "complete CV search tables", "test predictions for every model",
        "saved trained pipelines", "confusion matrices", "model comparison graph",
        "sampled error analysis", "false-positive/false-negative counts",
        "preprocessing experiment", "Random Forest seed stability",
        "McNemar significance tests"
    ]:
        print(f"- {item}")


if __name__ == "__main__":
    main()