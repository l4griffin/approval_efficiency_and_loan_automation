# Approval Efficiency & Loan Automation

## 1. Project Overview

This project uses machine learning to predict loan defaults in rapid-disbursement digital mobile lending platforms. The goal is to address the high rates of loan defaults and systemic bottlenecks—such as review fatigue and manual screening inefficiencies—by leveraging high-precision algorithmic screening and predictive scoring. 

The project covers data preprocessing, text parsing, model training, evaluation, and practical recommendations for financial lending institutions.

## 2. Project Objectives

* Explore and understand the hybrid financial and demographic dataset.
* Parse loan applications using Natural Language Processing (NLP) to map historical records to live loan requirements.
* Preprocess data for machine learning (handling missing values, dropping identifiers, one-hot encoding, and scaling).
* Train and compare four classification models to predict default risks.
* Evaluate model performance, highlighting the trade-offs between predictive accuracy and model explainability.

## 3. Dataset Description

The project utilizes a hybrid dataset containing 19 columns. The `african_context_digital_credit_data.csv` was used as the initial dataset, with code from `dataset generation.py` used to come up with missing features. Following this, the code from `dataset merge.py` was used to come up with `Merged_Dataset.csv`.

| Feature Profile | Description |
| :--- | :--- |
| **Demographic & Household** | `country`, `location_type` (Rural vs. Urban), `age_of_respondent`, `gender_of_respondent`, `marital_status`, `household_size`, `relationship_with_head`, `education_level`, `job_type`. |
| **Financial Access** | `bank_account`, `cellphone_access`. |
| **Digital Credit Metrics** | `Monthly_Income`, `Mobile_Money_Tx_Volume`, `Loan_Amount`, `Past_Defaults`. |
| **Target Variable** | `Default_Status` (1 = Default, 0 = Repaid). |

**Preprocessing Steps:** Rows with missing target variables are dropped. Non-predictive identifiers (`country`, `year`, `uniqueid`) are removed. Categorical variables are one-hot encoded, and numerical features are standardized using `StandardScaler`.

## 4. Machine Learning Models

Four classification algorithms are implemented to evaluate their predictive power on the dataset:
* **K-Nearest Neighbors (KNN):** Predicts outcomes based on the 5 data points closest to it.
* **Decision Tree:** Creates a flowchart of decision rules to split the data.
* **Artificial Neural Network (ANN):** A multilayer perceptron with one hidden layer of 100 neurons, running for up to 500 iterations to learn complex patterns.
* **Support Vector Machine (SVM):** Attempts to draw a linear boundary (straight line) that best separates the default classes.

The models are compared using 2-fold cross-validation and accuracy.

## 5. Model Results

The models reported the following average 2-fold cross-validation accuracies:

| Model | Accuracy |
| :--- | :--- |
| **SVM** | 91.16% |
| **ANN** | 89.18% |
| **Decision Tree** | 86.88% |
| **KNN** | 86.18% |

The **Support Vector Machine (SVM)** performed best, achieving 91.16% accuracy, indicating a relatively clear linear boundary between demographics, transaction volumes, and defaults. The **Decision Tree** offered 100% explainability, which is vital for regulatory compliance, but suffered from overfitting. **KNN** performed the poorest, severely handicapped by the "curse of dimensionality" introduced by one-hot encoding categorical variables.

## 6. Technologies Used

* Python
* Pandas and NumPy
* Scikit-learn (StandardScaler, KNeighborsClassifier, DecisionTreeClassifier, MLPClassifier, SVC)
* Natural Language Processing (NLP) frameworks (for Intelligent Parsing)

## 7. How to Run the Project

1. Clone or download this repository.
2. Install the required libraries:
   `pip install pandas numpy scikit-learn`
3. Generate the required dataset by running `dataset generation.py` and subsequently running `dataset merge.py` to output the final `Merged_Dataset.csv` file.
4. Run `ml_implementation.py` to execute the data preparation and model pipeline, which generates the cross-validation accuracy results. Ensure the random seed is set to `random_state=42` for reproducible results as written in the code.

## 8. Limitations and Future Improvements

* **Model Scaling:** While highly accurate, the SVM model scales poorly; if the financial dataset grows to millions of rows, training will become incredibly slow.
* **Overfitting:** The Decision Tree was allowed to grow without a `max_depth` parameter, causing it to perfectly memorize the training data and lowering its generalized accuracy.
* **Explainability:** The ANN operates as a "black box," making it impossible to extract a simple, logical rule to explain why a specific user was flagged for a default. 
* **Dimensionality:** KNN struggled mathematically with distance calculations due to the high number of binary columns created by `pd.get_dummies`.
* **Future Improvements:** Future work should focus on setting hyperparameters (like `max_depth` for decision trees), addressing dimensionality reduction before using KNN, and implementing more scalable models for large datasets.

## 9. Disclaimer

This project is for educational and research purposes. While it explores solutions for predicting credit default risk, it is not a production-ready financial assessment tool. Lending institutions must ensure compliance with local financial regulations and consumer protection laws before deploying automated screening systems.