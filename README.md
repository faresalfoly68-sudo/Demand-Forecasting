# Demand Forecasting System

### Retail Sales Prediction using Machine Learning

Demand Forecasting System is an end-to-end Machine Learning project that predicts retail store sales using historical sales data.

The project compares multiple regression models, performs feature engineering and hyperparameter tuning, and deploys the trained model through both a Streamlit web application and a FastAPI REST API.

## Features

- Retail sales prediction
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Multiple regression models
- Hyperparameter tuning using GridSearchCV
- Model evaluation
- Feature importance analysis
- Trained model serialization
- Streamlit web interface
- FastAPI REST API

## Dataset

The project uses the **Rossmann Store Sales** dataset.

The dataset contains more than **1 million retail sales records** and includes information related to:

- Store
- Sales
- Customers
- Promotions
- Holidays
- Competition
- Store information

### Dataset Download

The dataset is available on Google Drive:

[Download Dataset](https://drive.google.com/drive/folders/12Wq2aXkegkxxj3uF9jvoKwQBk6VkeQFY?usp=sharing)

### Main Features

- Day of Week
- Customers
- Promotion
- School Holiday
- Year
- Month
- Promotion-related features
- Competition duration

## Machine Learning Pipeline

```text
Data Loading
      ↓
Exploratory Data Analysis
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
Data Preprocessing
      ↓
Train / Test Split
      ↓
Model Training
      ↓
Hyperparameter Tuning
      ↓
Model Evaluation
      ↓
Model Selection
      ↓
Model Deployment
Models

The following regression models were evaluated:

Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor
Hyperparameter Tuning

GridSearchCV was used to tune model hyperparameters and improve model performance.

Results

The models were evaluated using:

R² Score
RMSE
MAE
Best Model

Gradient Boosting Regressor

Metric	Result
R²	~82%
RMSE	~2%
MAE	~1.4%

These results represent the evaluation results obtained during the project experiments.

Feature Importance

Feature importance analysis was performed to understand which features contributed most to the model's predictions.

This helps provide additional insight into the factors influencing retail sales predictions.

Project Structure
Demand-Forecasting/
│
├── Notebook/
│   ├── demand_forecasting_model.ipynb
│   └── demand_forecasting_model (2).ipynb
│
├── API_app.py
├── GUI_app.py
├── demand_forecasting_model.pkl
├── Demand.pdf
├── requirements.txt
├── .gitignore
└── README.md
Installation
git clone https://github.com/faresalfoly68-sudo/Demand-Forecasting.git
cd Demand-Forecasting

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
Run the Streamlit Application
streamlit run GUI_app.py

Then open the Streamlit URL displayed in the terminal.

Run the FastAPI Application
uvicorn API_app:app --reload

The API will be available at:

http://127.0.0.1:8000
API Documentation

FastAPI provides interactive API documentation at:

http://127.0.0.1:8000/docs
Technologies

Python • Pandas • NumPy • Matplotlib • Seaborn • Plotly • Scikit-learn • XGBoost • Joblib • Streamlit • FastAPI • Uvicorn

Model Deployment

The trained Machine Learning model is saved using Joblib and integrated into:

Streamlit application for interactive predictions
FastAPI REST API for programmatic predictions
Author

Fares Waleed Alfoly

AI / Machine Learning Engineer
