from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)

# Scale the features to improve convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Increase max_iter to allow more iterations for convergence
clf = LogisticRegression(random_state=808, max_iter=1000).fit(X_scaled, y)

print(clf.predict([X_scaled[8, :]])) # On obtient 0 : bénigne

print(clf.predict([X[13, :]]))

print(clf.predict_proba([X[8, :]]))

y_hat_proba = clf.predict_proba(X_scaled)[:,1]
import seaborn as sns
sns.histplot(y_hat_proba)
