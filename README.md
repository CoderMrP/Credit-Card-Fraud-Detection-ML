Credit Card Fraud Detection using Machine Learning

📌 Project Overview
This project focuses on detecting fraudulent credit card transactions using supervised machine learning techniques.
The objective is to build reliable classification models that can effectively identify fraud while handling severe class imbalance—a common real-world challenge in financial datasets.
The project evaluates multiple algorithms and prioritizes recall, ensuring fraudulent transactions are detected as accurately as possible from a business perspective.

🛠 Tech Stack
Programming Language: Python
Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
ML Techniques:
Logistic Regression
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Evaluation Metrics:
Recall
Precision
F1 Score
ROC-AUC
Tools: Jupyter Notebook, Google Colab

📊 Dataset
Source: Kaggle – Credit Card Fraud Detection Dataset
Description:
Transactions made by European cardholders
Features are anonymized using Principal Component Analysis (PCA)
Highly imbalanced dataset with very few fraud cases

🔄 Data Preprocessing
Exploratory Data Analysis (EDA)
Handling missing values
Feature scaling
Addressing class imbalance using:
SMOTE (oversampling)
Undersampling (final approach for realistic business impact)

🤖 Model Training & Evaluation
Trained and compared multiple models:
Logistic Regression
SVM
KNN
Logistic Regression performed best overall
Increased iterations to ensure convergence
Emphasis placed on recall to minimize missed fraud cases

📈 Key Insights
Fraudulent transactions are largely time-independent
Linear models perform exceptionally well due to data separability
Logistic Regression achieved strong recall while maintaining balance across metrics
Prioritizing recall is critical to prevent financial losses

📂 Project Structure
├── credit_card_fraud_detection_project.ipynb
├── raw_code/
│   └── experimental_models.ipynb
└── README.md

✅ Outcome
Successfully built a robust fraud detection model that:
Handles real-world class imbalance
Maximizes fraud detection recall
Demonstrates practical application of machine learning in finance

👤 Author
Paras Saini
