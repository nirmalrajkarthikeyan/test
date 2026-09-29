import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
hours = [1, 2, 3, 4, 5, 6, 7, 8]

scores = [35, 42, 50, 58, 65, 72, 80, 88]


X = np.array(hours).reshape(-1, 1)
y = np.array(scores)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()

model.fit(X_train, y_train)

prediction = model.predict([[10]])

print("Predicted score:", prediction)
