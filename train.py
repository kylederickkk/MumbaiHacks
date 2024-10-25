import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle

train_df = pd.read_csv('/carbon_neutral_budget_dataset.csv')

X = train_df.drop(['Budget Needed (INR)', 'Carbon Credits Required'], axis=1)
y_budget = train_df['Budget Needed (INR)']  # Target for budget prediction
y_credits = train_df['Carbon Credits Required']  # Target for carbon credits prediction

X_train, X_val, y_budget_train, y_budget_val, y_credits_train, y_credits_val = train_test_split(
    X, y_budget, y_credits, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

model_budget = LinearRegression()
model_budget.fit(X_train_scaled, y_budget_train)

model_credits = RandomForestRegressor(n_estimators=100, random_state=42)
model_credits.fit(X_train_scaled, y_credits_train)

with open('budget_model.pkl', 'wb') as f:
    pickle.dump(model_budget, f)

with open('credits_model.pkl', 'wb') as f:
    pickle.dump(model_credits, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Models and scaler saved successfully.")
