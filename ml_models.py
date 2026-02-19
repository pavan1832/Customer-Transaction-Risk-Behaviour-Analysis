"""
ml_models.py
Trains Logistic Regression and Random Forest classifiers,
evaluates them, and exposes a predict_single() utility.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble       import RandomForestClassifier
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics        import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "age", "amount", "credit_score",
    "account_tenure_months", "transaction_frequency",
    "cat_encoded",
]

def _prepare_features(df: pd.DataFrame):
    """Encode categoricals and return X, y, feature names."""
    tmp = df.copy()

    le = LabelEncoder()
    tmp["cat_encoded"] = le.fit_transform(tmp["category"])

    X = tmp[FEATURE_COLS].values.astype(float)
    y = tmp["is_high_risk"].values
    feature_names = [
        "Age", "Amount", "Credit Score",
        "Account Tenure (mo)", "Txn Frequency", "Category (enc)",
    ]
    return X, y, feature_names, le


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame) -> dict:
    """
    Train LR and RF classifiers.
    Returns a bundle dict with models, metrics, ROC data, feature importances,
    confusion matrices, scaler, and label encoder.
    """
    X, y, feature_names, le = _prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale for Logistic Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Model definitions ────────────────────────────────────────────────────
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=500, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42,
            class_weight="balanced", n_jobs=-1,
        ),
    }

    trained  = {}
    metrics_rows = []
    roc_data = []
    cms      = []

    for name, model in models.items():
        if name == "Logistic Regression":
            model.fit(X_train_sc, y_train)
            y_pred  = model.predict(X_test_sc)
            y_prob  = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred  = model.predict(X_test)
            y_prob  = model.predict_proba(X_test)[:, 1]

        trained[name] = model

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        auc  = roc_auc_score(y_test, y_prob)

        metrics_rows.append({
            "Model": name,
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1":        round(f1,   4),
            "ROC-AUC":   round(auc,  4),
        })

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append((name, fpr, tpr, auc))

        cms.append((name, confusion_matrix(y_test, y_pred)))

    # Feature importance from RF
    rf = trained["Random Forest"]
    fi = pd.DataFrame({
        "feature":    feature_names,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {
        "models":               trained,
        "scaler":               scaler,
        "label_encoder":        le,
        "feature_names":        feature_names,
        "metrics":              pd.DataFrame(metrics_rows).set_index("Model"),
        "roc_data":             roc_data,
        "feature_importance":   fi,
        "confusion_matrices":   cms,
        "categories":           list(le.classes_),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Live Prediction
# ─────────────────────────────────────────────────────────────────────────────

def predict_single(input_row: dict, bundle: dict, df: pd.DataFrame) -> list:
    """
    Predict risk for a single customer input.
    Returns list of (model_name, predicted_label, risk_probability).
    """
    le      = bundle["label_encoder"]
    scaler  = bundle["scaler"]
    models  = bundle["models"]

    # Encode category — handle unseen labels gracefully
    cat = input_row["category"]
    if cat in le.classes_:
        cat_enc = le.transform([cat])[0]
    else:
        cat_enc = 0  # fallback

    x = np.array([[
        input_row["age"],
        input_row["amount"],
        input_row["credit_score"],
        input_row["account_tenure_months"],
        input_row["transaction_frequency"],
        cat_enc,
    ]], dtype=float)

    x_sc = scaler.transform(x)

    results = []
    for name, model in models.items():
        x_input = x_sc if name == "Logistic Regression" else x
        prob    = model.predict_proba(x_input)[0, 1]
        label   = int(prob >= 0.5)
        results.append((name, label, prob))

    return results
