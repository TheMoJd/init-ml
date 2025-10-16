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

# pour améliroer les résultats on va faire la régression polynomiale

df['tv2'] = df.tv**2    

print(df)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df) # rends les valeures entre 0 et 1
data_array = scaler.transform(df)
df = pd.DataFrame(data_array, columns = ['tv','radio','journaux','ventes','tv2'])
print(df)
print(df.describe().loc[['min','max']])
X = df[['tv','radio','journaux', 'tv2']]
y = df.ventes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

reg.fit(X_train, y_train)
y_hat_test = reg.predict(X_test)

print(f"Coefficients: {reg.coef_}")
print(f"RMSE: {mean_squared_error(y_test, y_hat_test)}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_hat_test)}")