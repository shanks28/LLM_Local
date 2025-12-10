import numpy as np
import pandas as pd


def generate_loan_row(n):
    rng = np.random.default_rng(seed=42)
    # Categorical pools
    genders = ["Male", "Female"]
    married = ["Yes", "No"]
    dependents = ["0", "1", "2", "3+"]
    education = ["Graduate", "Not Graduate"]
    self_employed = ["Yes", "No"]
    property_area = ["Urban", "Semiurban", "Rural"]

    rows = []
    for i in range(n):
        loan_id = f"LP{i+1:06d}"
        gender = rng.choice(genders, p=[0.65, 0.35])
        married_ = rng.choice(married, p=[0.7, 0.3])
        depend = rng.choice(dependents, p=[0.6, 0.15, 0.15, 0.10])
        education_ = rng.choice(education, p=[0.8, 0.2])
        self_emp = rng.choice(self_employed, p=[0.12, 0.88])
        applicant_income = max(500, int(rng.normal(5000, 2500)))
        # coapplicant income often zero
        coapplicant_income = int(
            rng.choice([0, rng.normal(2000, 1500)], p=[0.6, 0.4])
        )
        # loan amount correlated with income
        loan_amount = max(
            20,
            int(np.clip(
                rng.normal(
                    applicant_income * 0.2 + coapplicant_income * 0.1, 50
                ),
                20, 600
            ))
        )
        loan_amount_term = int(
            rng.choice([120, 180, 240, 360], p=[0.05, 0.1, 0.15, 0.7])
        )
        # credit history 1 = good (more likely), 0 = bad
        credit_history = rng.choice([1, 0], p=[0.85, 0.15])
        property_area_ = rng.choice(property_area, p=[0.4, 0.35, 0.25])

        # Make label with some realistic logic
        # Higher income, credit_history=1, small loan -> higher approval
        # probability
        score = 0.0
        score += (applicant_income / 8000)  # scaled
        score += (coapplicant_income / 8000)
        score += (1.0 if credit_history == 1 else -0.6)
        score += (1.0 if education_ == "Graduate" else -0.1)
        score += (-(loan_amount / 600))
        # slight penalty for self_employed
        score += (-0.25 if self_emp == "Yes" else 0.0)
        prob = 1/(1+np.exp(-4*(score-0.5)))  # logistic scaling
        loan_status = "Y" if rng.random() < prob else "N"

        rows.append({
            "loan_id": loan_id,
            "gender": gender,
            "married": married_,
            "dependents": depend,
            "education": education_,
            "self_employed": self_emp,
            "applicant_income": applicant_income,
            "coapplicant_income": coapplicant_income,
            "loan_amount": loan_amount,
            "loan_amount_term": loan_amount_term,
            "credit_history": credit_history,
            "property_area": property_area_,
            "loan_status": loan_status
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_rows",
        type=int,
        default=100_000,
        help="number of rows to generate"
    )
    parser.add_argument("--out", type=str, default="loan_data_big.csv")
    args = parser.parse_args()
    df = generate_loan_row(args.n_rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
