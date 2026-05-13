# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the Iris dataset using load_iris().

2.Scale the input features using StandardScaler() for better SGD performance.

3.Split the dataset into training and testing sets using train_test_split().

4.Create and train the SGDClassifier model using the training data.

5.Predict the test data results and evaluate the model using accuracy score and classification report.


## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Mithun Kumar V
RegisterNumber:  212225040236
*/



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
iris = load_iris()
X = iris.data
y = iris.target
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = SGDClassifier(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

```

## Output:
<img width="933" height="315" alt="image" src="https://github.com/user-attachments/assets/c590e59b-c0c9-449b-8a6a-63f00f611b32" />


## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
