# train.py
import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.tree import export_text
from model import build_model

def train_rf(X_train, y_train, X_test, y_test, preprocessor, out_dir):
    rf = build_model()

    print("\n=== TRAINING RANDOM FOREST ===")
    rf.fit(X_train, y_train)  # verbose output appears here

    print("\n=== TRAINING COMPLETE ===")

    print("\n=== FEATURE IMPORTANCES ===")
    importances = rf.feature_importances_
    for idx in np.argsort(importances)[::-1][:10]:
        print(f"Feature {idx} importance: {importances[idx]:.4f}")

    print("\n=== SAMPLE TREE STRUCTURE (Tree 0) ===")
    tree_text = export_text(
        rf.estimators_[0],
        max_depth=3
    )
    print(tree_text)

    print("\n=== TEST SET PERFORMANCE ===")
    preds = rf.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(rf, f"{out_dir}/rf_model.joblib")
    joblib.dump(preprocessor, f"{out_dir}/preprocessor.joblib")

    print(f"\nModel saved to {out_dir}/")
