from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ==========================================
# RANDOM FOREST
# ==========================================

def train_random_forest(X_train, y_train):
    """
    Train Random Forest model.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_random_forest(model, X_test, y_test):
    """
    Evaluate Random Forest model.
    """

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision_macro = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall_macro = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Print results
    print()
    print("=" * 40)
    print("RANDOM FOREST")
    print("=" * 40)

    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Precision Macro: {precision_macro:.4f}")
    print(f"Recall Macro   : {recall_macro:.4f}")
    print(f"F1 Macro       : {f1_macro:.4f}")
    print(f"F1 Weighted    : {f1_weighted:.4f}")

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def main():
    """Run a small demo so this module can be tested directly."""

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=12,
        n_redundant=3,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = train_random_forest(X_train, y_train)
    evaluate_random_forest(model, X_test, y_test)


if __name__ == "__main__":
    main()