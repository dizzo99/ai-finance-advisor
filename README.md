# ai-finance-advisor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

class InvestmentPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()

    def train(self, data):
        # Assume data is a pandas DataFrame with columns:
        # 'age', 'income', 'risk_tolerance', 'investment_horizon', 'return'
        X = data[['age', 'income', 'risk_tolerance', 'investment_horizon']]
        y = data['return']

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)

    def predict(self, age, income, risk_tolerance, investment_horizon):
        input_data = np.array([[age, income, risk_tolerance, investment_horizon]])
        input_scaled = self.scaler.transform(input_data)
        predicted_return = self.model.predict(input_scaled)[0]

        # Generate investment suggestions based on the predicted return
        if predicted_return > 10:
            suggestion = "High-growth stocks and aggressive mutual funds"
        elif predicted_return > 7:
            suggestion = "Balanced portfolio of stocks and bonds"
        elif predicted_return > 4:
            suggestion = "Conservative mix of bonds and dividend stocks"
        else:
            suggestion = "Low-risk bonds and high-yield savings accounts"

        return {
            "predicted_return": predicted_return,
            "suggestion": suggestion
        }

# Example usage:
# predictor = InvestmentPredictor()
# data = pd.read_csv('investment_data.csv')
# predictor.train(data)
# result = predictor.predict(30, 75000, 7, 10)
# print(result)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from investment_model import InvestmentPredictor
from saving_model import SavingStrategyRecommender
from plaid import Client as PlaidClient
from forex_python.converter import CurrencyRates

app = FastAPI()

# Initialize models and clients
investment_predictor = InvestmentPredictor()
saving_recommender = SavingStrategyRecommender()
plaid_client = PlaidClient(client_id='your_plaid_client_id', secret='your_plaid_secret', environment='sandbox')
currency_converter = CurrencyRates()

# Load and preprocess data (in a real-world scenario, this would be more complex)
investment_data = pd.read_csv('investment_data.csv')
saving_data = pd.read_csv('saving_data.csv')

investment_predictor.train(investment_data)
saving_recommender.train(saving_data)

class InvestmentRequest(BaseModel):
    age: int
    income: float
    risk_tolerance: int
    investment_horizon: int

class SavingRequest(BaseModel):
    income: float
    expenses: float
    savings_rate: float
    debt: float

class PlaidLinkTokenRequest(BaseModel):
    user_id: str

class PlaidExchangeTokenRequest(BaseModel):
    public_token: str
    user_id: str

class TransactionRequest(BaseModel):
    access_token: str
    start_date: str
    end_date: str
    base_currency: str

@app.post("/create_link_token")
async def create_link_token(request: PlaidLinkTokenRequest):
    try:
        response = plaid_client.link_token_create({
            'user': {
                'client_user_id': request.user_id,
            },
            'products': ['transactions'],
            'client_name': 'AI Finance Advisor',
            'country_codes': ['US'],
            'language': 'en'
        })
        return {"link_token": response['link_token']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/exchange_public_token")
async def exchange_public_token(request: PlaidExchangeTokenRequest):
    try:
        exchange_response = plaid_client.item_public_token_exchange(request.public_token)
        access_token = exchange_response['access_token']
        # In a real app, you'd securely store this access_token associated with the user
        return {"access_token": access_token}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/get_transactions")
async def get_transactions(request: TransactionRequest):
    try:
        response = plaid_client.transactions_get(
            request.access_token,
            start_date=request.start_date,
            end_date=request.end_date
        )
        transactions = response['transactions']
        
        # Convert all transactions to the base currency
        for transaction in transactions:
            if transaction['iso_currency_code'] != request.base_currency:
                original_amount = transaction['amount']
                original_currency = transaction['iso_currency_code']
                converted_amount = currency_converter.convert(original_currency, request.base_currency, original_amount)
                transaction['original_amount'] = original_amount
                transaction['original_currency'] = original_currency
                transaction['amount'] = converted_amount
                transaction['iso_currency_code'] = request.base_currency

        return {"transactions": transactions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_investment")
async def predict_investment(request: InvestmentRequest):
    try:
        result = investment_predictor.predict(
            request.age,
            request.income,
            request.risk_tolerance,
            request.investment_horizon
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend_saving_strategy")
async def recommend_saving_strategy(request: SavingRequest):
    try:
        result = saving_recommender.recommend(
            request.income,
            request.expenses,
            request.savings_rate,
            request.debt
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class SavingStrategyRecommender:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        self.scaler = StandardScaler()

    def train(self, data):
        # Assume data is a pandas DataFrame with columns:
        # 'income', 'expenses', 'savings_rate', 'debt'
        X = data[['income', 'expenses', 'savings_rate', 'debt']]
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)

    def recommend(self, income, expenses, savings_rate, debt):
        input_data = np.array([[income, expenses, savings_rate, debt]])
        input_scaled = self.scaler.transform(input_data)
        cluster = self.kmeans.predict(input_scaled)[0]

        strategies = [
            [
                "Increase your emergency fund",
                "Look for ways to reduce high-interest debt",
                "Consider a side hustle to boost income"
            ],
            [
                "Maximize contributions to retirement accounts",
                "Diversify your investment portfolio",
                "Set specific financial goals and create a plan to achieve them"
            ],
            [
                "Focus on increasing your savings rate",
                "Review and cut unnecessary expenses",
                "Automate your savings to ensure consistency"
            ],
            [
                "Prioritize paying off high-interest debt",
                "Create a strict budget to control spending",
                "Explore opportunities for career advancement or additional education"
            ]
        ]

        return {
            "cluster": cluster,
            "strategies": strategies[cluster]
        }

# Example usage:
# recommender = SavingStrategyRecommender()
# data = pd.read_csv('saving_data.csv')
# recommender.train(data)
# result = recommender.recommend(75000, 60000, 0.2, 10000)
# print(result)

