# predict.py
import joblib
import random


def predict_random(X_test, y_test, artifacts_dir):
    rf = joblib.load(f"{artifacts_dir}/rf_model.joblib")

    idx = random.randint(0, X_test.shape[0] - 1)
    sample = X_test[idx].reshape(1, -1)
    prob = rf.predict_proba(sample)[0][1]
    pred = rf.predict(sample)[0]

    print("\n=== PREDICTION ===")
    print(f"Test index        : {idx}")
    print(f"Approval prob     : {prob:.4f}")
    print(f"Predicted result  : {'APPROVED' if pred == 1 else 'REJECTED'}")
    actual = 'APPROVED' if y_test.iloc[idx] == 1 else 'REJECTED'
    print(f"Actual label      : {actual}")
