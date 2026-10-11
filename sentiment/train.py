"""
Reproducible training and evaluation script for CFI Sentiment Classifier.
Trains a TF-IDF + LogisticRegression baseline pipeline on a multi-domain feedback dataset.
Generates evaluation metrics (accuracy, precision, recall, F1, confusion matrix) and persists artifacts.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sentiment.preprocessing import clean_text

MODEL_VERSION = "v1.0-tfidf-logreg"
MODEL_FILENAME = f"sentiment_model_{MODEL_VERSION}.joblib"
METADATA_FILENAME = f"model_metadata_{MODEL_VERSION}.json"


def get_labeled_training_corpus():
    """
    Returns a balanced multi-domain customer feedback corpus (Positive, Neutral, Negative).
    Covers retail, technology, logistics, customer service, and food/hospitality.
    """
    # Base templates and phrases across domains
    positive_phrases = [
        "Absolutely amazing product! Exceeded all my expectations.",
        "The customer service was remarkably helpful and resolved my issue in minutes.",
        "High quality materials, arrived on time, and works like a charm.",
        "Best purchase I have made all year. Highly recommended to everyone!",
        "Very intuitive user interface and super fast response times.",
        "Fantastic experience from ordering to unboxing. Five stars!",
        "The battery life is outstanding and easily lasts two full days.",
        "Delicious food, courteous staff, and wonderful atmosphere.",
        "Delivery was surprisingly quick and packaging was top notch.",
        "Great value for money. Performs even better than much more expensive alternatives.",
        "Extremely reliable and easy to configure. Saved our team dozens of hours.",
        "Brilliant engineering and thoughtful design. Couldn't be happier.",
        "Smooth return process and prompt refund. Trustworthy seller.",
        "Super comfortable and fits true to size. Will definitely order again.",
        "The new update fixed all previous bugs and brought excellent new features.",
        "Customer support agent went above and beyond to help me out.",
        "Remarkably crisp display and loud, clear speakers. Love it!",
        "Works seamlessly with all my other devices. A joy to use.",
        "Prompt shipment, accurate description, and friendly communication.",
        "Impressed with the durability and build quality after three months of daily use.",
        "Wonderful experience, seamless checkout, and very fast shipping.",
        "Exemplary service and top tier product quality. A+",
        "Clean, modern, and very efficient. Thoroughly satisfied with this purchase.",
        "The sound quality is crystal clear with deep bass.",
        "Flawless performance, no stutter or lag whatsoever.",
        "Outstanding customer care! They replaced my damaged unit without any hassle.",
        "I am thoroughly impressed by the speed and accuracy of this service.",
        "Sturdy build, easy setup instructions, and works exactly as advertised.",
        "Delivered a day earlier than scheduled. Perfect condition!",
        "Worth every single penny. Would buy again without hesitation.",
    ]

    neutral_phrases = [
        "The product arrived as described. Nothing particularly special or bad.",
        "Average experience overall. It does the job but lacks standout features.",
        "Standard delivery time, package arrived without issues.",
        "The item is acceptable for the price point, but build quality is basic.",
        "It functions as specified in the manual. Typical performance.",
        "Neutral impression so far. I will need to test it further over time.",
        "It is okay, neither good nor bad. Does what it says on the box.",
        "The interface is functional though slightly dated.",
        "Delivery took exactly the estimated five business days.",
        "Customer support answered my questions with standard boilerplate responses.",
        "The color matches the photo, though sizing is slightly loose.",
        "Moderate battery life. Lasts about one day of normal usage.",
        "Expected performance given the discount price.",
        "Has both pros and cons. Good screen but mediocre speakers.",
        "It works as expected. No real complaints, no real praise.",
        "Setup was neither difficult nor especially intuitive. Took about twenty minutes.",
        "Adequate for light home use, but probably not suited for heavy commercial tasks.",
        "Received standard automated confirmation emails.",
        "The fabric feels ordinary, comparable to other items in this price tier.",
        "The refund took seven business days as noted in their terms.",
        "Average build quality. It is plastic but feels sturdy enough.",
        "Packaging was basic brown cardboard without extra cushioning.",
        "Works fine for now, will see how it holds up over the coming months.",
        "The software gets the job done but navigation could be slightly cleaner.",
        "Acceptable sound quality for casual listening.",
        "Customer service responded within 24 hours with standard instructions.",
        "Standard replacement parts, fit properly into the slot.",
        "Delivery arrived on the last day of the promised window.",
        "Fair price for what is included in the package.",
        "It meets basic requirements, though there are alternatives with better aesthetics.",
    ]

    negative_phrases = [
        "Terrible experience. The item arrived broken and unusable.",
        "Customer support was completely unhelpful and ignored my emails.",
        "Cheap plastic build, broke after only two days of gentle use.",
        "Delivery was delayed by two weeks with zero communication from the seller.",
        "Extremely disappointing quality. Nothing like what was shown in the pictures.",
        "Constantly crashing and full of bugs. Completely wasted my money.",
        "The device overheats within ten minutes of turning it on. Avoid this!",
        "Unacceptable delay and missing parts in the package.",
        "Rude support staff refused to honor the warranty policy.",
        "Worst purchase ever. Completely regret buying this item.",
        "Battery drains from 100% to zero in less than two hours. Defective unit.",
        "Misleading product description. Does not support the advertised features.",
        "Fell apart immediately upon opening the package. Absolute junk.",
        "Slow, unresponsive, and constantly freezes. Do not recommend.",
        "Refund was delayed for over a month despite multiple follow-up calls.",
        "The zipper broke on the first wear. Horrible craftsmanship.",
        "Severe lag and glitchy performance. Practically unusable software.",
        "Package arrived crushed and soaking wet. Items inside were ruined.",
        "Very frustrating setup process with missing documentation.",
        "Overpriced and underperforming. You can find much better options elsewhere.",
        "Customer care hung up on me twice. Inexcusable lack of professionalism.",
        "Stopped working completely after one week. Waste of hard-earned money.",
        "Screen has dead pixels right out of the box. Terrible quality control.",
        "Shrank two sizes after one gentle wash following manufacturer care instructions.",
        "App keeps failing to sync data and losing my work.",
        "Missing essential accessories that were supposed to be included in the box.",
        "Very poor audio quality with constant background buzzing and static.",
        "Deceptive marketing! Charged extra hidden fees that were not disclosed.",
        "Completely dissatisfied with this purchase. Demanding a full refund.",
        "The software update completely broke the core functionality.",
    ]

    # Modifiers and domain contexts to generate a robust, diverse dataset of 1,500+ records
    contexts = [
        "",
        "For my home office setup: ",
        "Regarding my recent order: ",
        "After using it for two weeks: ",
        "In terms of software and usability: ",
        "Comparing with my previous brand: ",
        "Speaking about the hardware and durability: ",
        "My feedback on the customer service team: ",
        "From the perspective of a daily user: ",
        "Honest review after one month: ",
        "Shipping and fulfillment review: ",
        "Packaging and unboxing experience: ",
        "Value assessment: ",
        "Mobile app performance: ",
        "Return policy experience: ",
        "Tech support interaction: ",
        "Build quality and finishing: ",
    ]

    texts = []
    labels = []

    for prefix in contexts:
        for p in positive_phrases:
            texts.append(clean_text(f"{prefix}{p}"))
            labels.append("POSITIVE")
        for n in neutral_phrases:
            texts.append(clean_text(f"{prefix}{n}"))
            labels.append("NEUTRAL")
        for neg in negative_phrases:
            texts.append(clean_text(f"{prefix}{neg}"))
            labels.append("NEGATIVE")

    return texts, labels


def train_and_evaluate(artifacts_dir=None):
    """
    Train TF-IDF + Logistic Regression pipeline, evaluate, and save artifacts.
    """
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).resolve().parent.parent / "ml_artifacts"
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print("Loading labeled customer feedback corpus...")
    texts, labels = get_labeled_training_corpus()
    total_samples = len(texts)
    print(f"Total labeled samples: {total_samples}")

    # Stratified train/test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.20,
        random_state=42,
        stratify=labels,
    )
    print(f"Training set: {len(X_train)} | Test set: {len(X_test)}")

    # Define model pipeline: TF-IDF vectorizer + Logistic Regression
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    max_features=12000,
                    sublinear_tf=True,
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    print("Fitting model pipeline on training data...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model on held-out test data...")
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    class_labels = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
    accuracy = float(accuracy_score(y_test, y_pred))
    report_dict = classification_report(
        y_test, y_pred, target_names=class_labels, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred, labels=class_labels).tolist()

    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_labels))

    # Save model artifact
    model_path = artifacts_dir / MODEL_FILENAME
    joblib.dump(pipeline, model_path)
    print(f"Saved model artifact to: {model_path}")

    # Metadata record
    metadata = {
        "model_version": MODEL_VERSION,
        "model_type": "TF-IDF (1-2 ngrams) + Logistic Regression (balanced)",
        "trained_at": datetime.now(UTC).isoformat(),
        "sample_count": total_samples,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "classes": class_labels,
        "evaluation_metrics": {
            "accuracy": round(accuracy, 4),
            "macro_precision": round(report_dict["macro avg"]["precision"], 4),
            "macro_recall": round(report_dict["macro avg"]["recall"], 4),
            "macro_f1": round(report_dict["macro avg"]["f1-score"], 4),
            "weighted_f1": round(report_dict["weighted avg"]["f1-score"], 4),
            "per_class": {
                cls_name: {
                    "precision": round(report_dict[cls_name]["precision"], 4),
                    "recall": round(report_dict[cls_name]["recall"], 4),
                    "f1_score": round(report_dict[cls_name]["f1-score"], 4),
                    "support": report_dict[cls_name]["support"],
                }
                for cls_name in class_labels
            },
        },
        "confusion_matrix": {
            "labels": class_labels,
            "matrix": cm,
        },
        "hyperparameters": {
            "vectorizer_ngram_range": [1, 2],
            "vectorizer_max_features": 12000,
            "vectorizer_sublinear_tf": True,
            "classifier_C": 1.0,
            "classifier_class_weight": "balanced",
            "classifier_max_iter": 1000,
        },
        "known_limitations": [
            "Does not parse complex sarcasm or irony without explicit lexical markers.",
            "Domain-specific technical jargon may receive lower confidence.",
            "Short texts (<3 words) provide limited context for reliable neutral vs positive separation.",
            "Probabilities reflect softmax logistic regression outputs and should be interpreted as calibrated confidence estimates rather than absolute ground truth.",
        ],
    }

    metadata_path = artifacts_dir / METADATA_FILENAME
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {metadata_path}")

    return pipeline, metadata


if __name__ == "__main__":
    train_and_evaluate()
