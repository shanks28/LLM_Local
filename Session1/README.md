# Loan Approval Prediction Model

This project implements a Random Forest classifier to predict loan approval status based on applicant features such as income, loan amount, credit history, employment years, and dependents.

## Prerequisites

- Python 3.7+
- Required packages:
  - pandas
  - numpy
  - scikit-learn
  - joblib

Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

## Dataset Generation

Before training the model, you need to generate a dataset. Use the provided `generator_2.py` script to create a synthetic loan dataset.

### Generate Dataset

```bash
python generator_2.py
```

This will create `loan_data_v2.csv` in the current directory with 100,000 rows by default.

### Generator Parameters

You can modify the `generator_2.py` file to customize dataset generation:

- `n_rows`: Number of rows to generate (default: 100,000)
- `seed`: Random seed for reproducibility (default: 42)

Example with custom parameters:
```python
df = generate(n_rows=50_000, seed=123)
df.to_csv("loan_data_v2.csv", index=False)
```

### Dataset Format

The generated CSV file contains the following columns:
- `income`: Applicant income (1500-15000)
- `loan_amount`: Requested loan amount (50-600)
- `credit_history`: Credit history status (0 or 1)
- `employment_years`: Years of employment (0-15)
- `dependents`: Number of dependents (0-4)
- `loan_status`: Target variable ("Y" for approved, "N" for rejected)

## Model Training

Train the Random Forest model using the `main.py` script:

```bash
python main.py train --data loan_data_v2.csv --artifacts artifacts
```

### Training Parameters

- `--data`: Path to the CSV dataset file (required)
- `--artifacts`: Directory to save model artifacts (default: "artifacts")

The training process will:
1. Load and preprocess the data
2. Split into train/test sets (80/20 split)
3. Train a Random Forest classifier
4. Display feature importances
5. Show sample tree structure
6. Print classification report on test set
7. Save the trained model and preprocessor to the artifacts directory

### Model Configuration

The Random Forest model is configured in `model.py` with the following parameters:

- `n_estimators`: 100 (number of trees in the forest)
- `max_depth`: 10 (maximum depth of each tree)
- `min_samples_split`: 2 (minimum samples required to split a node)
- `random_state`: 42 (for reproducibility)
- `n_jobs`: -1 (use all available CPU cores)
- `verbose`: 1 (show training progress)

You can modify these parameters in `Session1/model.py` to tune the model performance.

## Model Testing/Prediction

After training, test the model on the test set:

```bash
python main.py test --data loan_data_v2.csv --artifacts artifacts
```

This will:
1. Load the trained model from the artifacts directory
2. Randomly select a sample from the test set
3. Display the prediction probability and result
4. Show the actual label for comparison

### Prediction Parameters

- `--data`: Path to the CSV dataset file (required)
- `--artifacts`: Directory containing saved model artifacts (default: "artifacts")

## Project Structure

```
Session1/
├── generator_2.py      # Dataset generation script
├── data.py             # Data loading and preprocessing
├── model.py            # Model definition
├── train.py            # Training logic
├── predict.py          # Prediction logic
├── main.py             # Main entry point
├── artifacts/          # Saved models and preprocessors
│   ├── rf_model.joblib
│   └── preprocessor.joblib
└── README.md           # This file
```

## Usage Examples

### Complete Workflow

1. **Generate dataset:**
   ```bash
   python generator_2.py
   ```

2. **Train the model:**
   ```bash
   python main.py train --data loan_data_v2.csv
   ```

3. **Test the model:**
   ```bash
   python main.py test --data loan_data_v2.csv
   ```

### Using Custom Artifacts Directory

```bash
python main.py train --data loan_data_v2.csv --artifacts my_models
python main.py test --data loan_data_v2.csv --artifacts my_models
```

## Model Output

During training, you'll see:
- Training progress (verbose output)
- Top 10 feature importances
- Sample tree structure (first tree, depth 3)
- Classification report with precision, recall, and F1-score

During testing, you'll see:
- Random test sample index
- Approval probability (0-1)
- Predicted result (APPROVED/REJECTED)
- Actual label (APPROVED/REJECTED)

## Notes

- The dataset uses rule-based logic with 5% noise for realistic training
- The model uses stratified train/test split to maintain class distribution
- All preprocessing (imputation, encoding) is saved with the model for consistent predictions
- The artifacts directory will be created automatically if it doesn't exist
