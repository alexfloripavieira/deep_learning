from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

perceptron = Perceptron(max_iter=100, eta0=0.1, random_state=42)

perceptron.fit(X_train, y_train)

y_pred = perceptron.predict(X_test)

print(accuracy_score(y_test, y_pred))
