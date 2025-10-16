import pandas as pd
df = pd.read_csv('datasets/advertising.csv')
# Explorons rapidement les données avec
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
# variables prédicteurs
X = df[['tv','radio','journaux']]
# cible
y = df.ventes
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 200 échantillons, 80% entrainements: 160, 20% tests: 40
#on entraine le modèle
reg.fit(X_train, y_train)

# Pour estimer la performance, on récupère les prédictions pour X_test
y_pred_test = reg.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
print(f"RMSE: {mean_squared_error(y_test, y_pred_test)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred_test)}")