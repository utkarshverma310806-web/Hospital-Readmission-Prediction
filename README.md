# Hospital-Readmission-Prediction
# Hospital Readmission Prediction

## 📌 Project Overview

This project predicts whether a patient is likely to be readmitted to the hospital within 30 days.

A **Logistic Regression model with L2 regularization** is used for prediction. The model is evaluated using **ROC-AUC**, confusion matrix, precision, recall, and F1-score.

## 📊 Dataset

The dataset contains 5,000 patient records for training and 2,000 records for testing.

### Features

* `age` — Patient age
* `gender` — Patient gender
* `primary_diagnosis` — Primary medical diagnosis
* `num_procedures` — Number of medical procedures
* `days_in_hospital` — Length of hospital stay
* `comorbidity_score` — Comorbidity score
* `discharge_to` — Patient's discharge destination

### Target

`readmitted`

* `0` → Not readmitted
* `1` → Readmitted

## 🔧 Methodology

### 1. Data Preprocessing

The dataset was divided into training and testing sets using an 80:20 split.

Numerical features were standardized using `StandardScaler`.

Categorical features were converted into numerical form using `OneHotEncoder`.

### 2. Model

Logistic Regression was used with **L2 regularization**.

```python
LogisticRegression(
    penalty="l2",
    C=1.0,
    max_iter=1000
)
```

### 3. Model Evaluation

The model was evaluated using:

* ROC-AUC
* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## 📈 Results

The model achieved:

| Metric    | Result |
| --------- | -----: |
| ROC-AUC   | 0.5067 |
| Accuracy  | 0.6840 |
| Precision | 0.2193 |
| Recall    | 0.2660 |
| F1-score  | 0.2405 |

The ROC-AUC of approximately **0.507** indicates that the model has limited ability to distinguish between readmitted and non-readmitted patients on the held-out test set.

## 🎯 Threshold Analysis

A probability threshold of `0.20` was used for the detailed confusion-matrix analysis.

The resulting confusion matrix was:

```text
                 Predicted
              0          1

Actual 0     634        178
Actual 1     138         50
```

### False Positive

A false positive occurs when the model predicts that a patient will be readmitted, but the patient is not actually readmitted.

Potential impact:

* Unnecessary follow-up
* Additional monitoring
* Extra use of hospital resources

### False Negative

A false negative occurs when the patient is actually readmitted, but the model fails to identify the patient as high risk.

Potential impact:

* A potentially high-risk patient may not receive additional monitoring or intervention
* An opportunity for preventive follow-up may be missed

In a clinical setting, the choice of classification threshold depends on the relative clinical and operational costs of false positives and false negatives.

## 📁 Project Files

```text
Hospital-Readmission-Prediction/
│
├── hospital_readmission.ipynb
├── README.md
└── submission.csv
```

## 🛠️ Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## ▶️ How to Run

1. Open `hospital_readmission.ipynb` in Google Colab.
2. Upload:

   * `train_df.csv`
   * `test_df.csv`
   * `sample_submission.csv`
3. Run the notebook cells sequentially.
4. The model will train and generate predictions.
5. The final predictions are saved as `submission.csv`.

## 📌 Conclusion

The Logistic Regression model successfully completed the required hospital readmission prediction pipeline, including preprocessing, L2 regularization, ROC-AUC evaluation, threshold analysis, and final test-set prediction.

The obtained ROC-AUC shows that the available features provide limited predictive separation for this dataset. Therefore, the result should be interpreted as a limitation of the current feature set/model rather than as evidence of strong clinical prediction performance.
