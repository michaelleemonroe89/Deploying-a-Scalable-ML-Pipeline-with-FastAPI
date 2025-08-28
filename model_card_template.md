# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a RandomForestClassifier trained to predict whether an individual's income exceeds $50K/year based on U.S. Census data. The model uses features such as workclass, education, marital status, occupation, relationship, race, sex, and native country.

- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Version:** 1.0
- **Author:** Mike Monroe
- **Date:** 08/27/2025

## Intended Use
The model is intended for educational and demonstration purposes, showcasing how to build, deploy, and evaluate a machine learning pipeline using FastAPI. It is not intended for production or high-stakes decision-making.

- **Primary intended users:** Data science students, educators, and developers.
- **Primary intended uses:** Demonstration of ML pipeline deployment and evaluation.

## Training Data
The model was trained on the "census.csv" dataset, which is a processed version of the UCI Adult dataset. The dataset contains demographic information and income labels.

- **Number of training samples:** ~80% of the full dataset (after train/test split)
- **Features used:** workclass, education, marital-status, occupation, relationship, race, sex, native-country

## Evaluation Data
The evaluation data consists of the remaining ~20% of the census dataset, held out from training.

- **Number of evaluation samples:** ~20% of the full dataset
- **Evaluation method:** Standard train/test split

## Metrics
_Please include the metrics used and your model's performance on those metrics._

- **Precision:** 0.7419
- **Recall:** 0.6384
- **F1 Score:** 0.6863

Metrics were also computed on slices of the data for each value of the categorical features to evaluate fairness and performance consistency.

## Ethical Considerations
- The model may reflect biases present in the original census data, including those related to race, gender, and nationality.
- Predictions should not be used for real-world decision-making without further bias and fairness analysis.
- Sensitive attributes are present in the data and may influence predictions.

## Caveats and Recommendations
- The model is for demonstration only and not suitable for production use.
- Further tuning, validation, and fairness analysis are recommended before any deployment in real-world scenarios.
- Users should be aware of potential biases and limitations in both the data and the model.
