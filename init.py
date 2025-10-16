import pandas as pd

# lecture du jeu de données
df = pd.read_csv('age_vs_poids_vs_taille_vs_sexe.csv')

# Les variables prédictives
X = df[['age', 'taille', 'sexe']]

# Variable cible: poids
y = df['poids']

# on choisit un modèle de régression linéaire
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# On entraine le modèle avec la méthode fit
reg.fit(X, y)

# et on obtient un score
print('R^2 score:', reg.score(X, y))

# ainsi que les coefficients a,b,c de la régression linéaire
print('Coefficients:', reg.coef_)
