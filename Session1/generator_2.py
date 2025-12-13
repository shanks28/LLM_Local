# generate_dataset_v2.py
import numpy as np
import pandas as pd


def generate(n_rows=100_000, seed=42):
    rng = np.random.default_rng(seed)
    rows = []

    for i in range(n_rows):
        income = rng.integers(1500, 15000)
        loan_amount = rng.integers(50, 600)
        credit_history = rng.choice([0, 1], p=[0.2, 0.8])
        employment_years = rng.integers(0, 15)
        dependents = rng.integers(0, 4)

        # Rule-based approval logic
        approved = 1

        if credit_history == 0:
            approved = 0
        if loan_amount > income * 0.35:
            approved = 0
        if employment_years < 2 and loan_amount > 200:
            approved = 0

        # small noise
        if rng.random() < 0.05:
            approved = 1 - approved

        rows.append({
            "income": income,
            "loan_amount": loan_amount,
            "credit_history": credit_history,
            "employment_years": employment_years,
            "dependents": dependents,
            "loan_status": "Y" if approved else "N"
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    df.to_csv("loan_data_v2.csv", index=False)
