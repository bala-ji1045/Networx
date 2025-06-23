# fastapi_app.py
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# 1. Define Pydantic Model for API Input
class UserInput(BaseModel):
    Age: int
    Gender: str
    Location: str
    Income: int
    Interests: str
    Total_Spending: int
    Product_Category_Preference: str
    Time_Spent_on_Site_Minutes: int

# 2. Load Data and Model Components
df = pd.read_csv("user_personalized_features.csv")
features = [
    'Age', 'Gender', 'Location', 'Income', 'Interests',
    'Total_Spending', 'Product_Category_Preference', 'Time_Spent_on_Site_Minutes'
]
X = df[features]

numeric_features = ['Age', 'Income', 'Total_Spending', 'Time_Spent_on_Site_Minutes']
categorical_features = ['Gender', 'Location', 'Interests', 'Product_Category_Preference']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
])
X_processed = preprocessor.fit_transform(X)
X_tensor = torch.tensor(X_processed, dtype=torch.float32)

# Define Autoencoder Model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=3):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

# Train Model or Load from File
model = Autoencoder(input_dim=X_tensor.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    output = model(X_tensor)
    loss = criterion(output, X_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

compressed_embeddings = model.encoder(X_tensor).detach().numpy()

# 3. Start FastAPI app
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["*"] for all origins (not recommended for production)Add commentMore actions
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/recommend")
def recommend(user: UserInput):
    user_df = pd.DataFrame([user.dict()])
    user_processed = preprocessor.transform(user_df)
    user_tensor = torch.tensor(user_processed, dtype=torch.float32)
    user_embedding = model.encoder(user_tensor).detach().numpy()

    similarities = cosine_similarity(user_embedding, compressed_embeddings)[0]
    top_indices = np.argsort(similarities)[-5:][::-1]

    recommendations = df.iloc[top_indices][['Name', 'User_ID', 'Age', 'Gender', 'Location', 'Interests']]
    recommendations['Similarity (%)'] = (similarities[top_indices] * 100).round(2)

    return recommendations.to_dict(orient="records")
