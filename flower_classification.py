import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)
df["species"] = iris.target

df["species"] = df["species"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())
print("\nFlower species count:")
print(df["species"].value_counts())

print("\nGenerating feature pair visualization...")

sns.pairplot(
    df,
    hue="species",
    diag_kind="hist"
)

plt.suptitle(
    "Iris Flower Feature Pair Visualization",
    y=1.02
)

plt.show()

X = df.drop("species", axis=1)
y = df["species"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data size:")
print(X_train.shape)

print("\nTesting data size:")
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logistic_model = LogisticRegression(max_iter=200)

logistic_model.fit(X_train, y_train)

y_pred_logistic = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, y_pred_logistic)

print("\nLogistic Regression Accuracy:")
print(f"{logistic_accuracy * 100:.2f}%")
from sklearn.tree import DecisionTreeClassifier

decision_tree_model = DecisionTreeClassifier(random_state=42)

decision_tree_model.fit(X_train, y_train)
y_pred_tree = decision_tree_model.predict(X_test)

tree_accuracy = accuracy_score(y_test, y_pred_tree)

print("\nDecision Tree Accuracy:")
print(f"{tree_accuracy * 100:.2f}%")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred_logistic)

print("\nConfusion Matrix - Logistic Regression:")
print(cm)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=logistic_model.classes_
)

disp.plot()

plt.title("Confusion Matrix - Logistic Regression")
plt.show()

print("\n--- New Flower Prediction ---")

sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))

new_flower = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

prediction = logistic_model.predict(new_flower)

print("\nPredicted flower species:")
print(prediction[0])