# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Collect the dataset
Obtain the employee dataset containing independent variables (features such as age, salary, job role, years at company, etc.) and the dependent variable Y (Employee Churn: Yes/No).

2.Identify variables
Let X be the set of independent variables (employee attributes).
Let Y be the dependent variable representing churn status.

3.Preprocess the data
Handle missing values if any.
Encode categorical variables into numerical form.
Split the dataset into training data and testing data.

4.Select the splitting criterion
Choose a measure to evaluate splits at each node:
Gini Index or
Entropy (Information Gain)
Entropy formula:
<img width="501" height="70" alt="image" src="https://github.com/user-attachments/assets/058e59ed-6454-4e59-ad9e-8b62ebabe87e" />

5.Find the best attribute for splitting
Calculate the Information Gain or Gini Index for each feature.
Select the attribute that provides the maximum Information Gain or minimum Gini Index.

6.Create the decision tree
Make the selected attribute the decision node.
Split the dataset into subsets based on attribute values.
Repeat steps 4–6 recursively for each subset.

7.Apply stopping conditions
Stop splitting when:
All instances belong to the same class, or
No attributes remain, or
Maximum tree depth is reached.

8.Assign class labels
Label the leaf nodes with the majority class (Churn = Yes or No).

9.Train the model
Construct the decision tree using the training dataset.

10.Test the model
Use the testing dataset to predict employee churn.

11.Evaluate model performance
Measure accuracy, precision, recall, or confusion matrix to assess performance.

12.Predict employee churn
Use the trained Decision Tree model to predict churn for new employee data.
## Program:
```python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: SUDARSAN.A
RegisterNumber:  212224220111
*/
```
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load dataset
data = pd.read_csv("/content/Employee.csv")

# Basic checks
print(data.head())
print(data.info())
print(data.isnull().sum())
print(data["left"].value_counts())

# Encode salary column
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])

# Select features
X = data[[
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary"
]]

y = data["left"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=100
)

# Create Decision Tree model (prevent overfitting)
dt = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=4,
    random_state=42
)

# Train model
dt.fit(X_train, y_train)

# Predictions
y_pred = dt.predict(X_test)

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
feature_importance = pd.Series(dt.feature_importances_, index=X.columns)
print("\nFeature Importance:")
print(feature_importance.sort_values(ascending=False))

# Example Prediction
sample_employee = [[0.5, 0.8, 9, 260, 6, 0, 1, 2]]
prediction = dt.predict(sample_employee)

if prediction[0] == 1:
    print("Prediction: Employee will leave")
else:
    print("Prediction: Employee will stay")

# Plot Decision Tree
plt.figure(figsize=(12,8))
plot_tree(
    dt,
    feature_names=X.columns,
    class_names=['Stayed', 'Left'],
    filled=True
)
plt.show()
```

## Output:
<img width="823" height="147" alt="image" src="https://github.com/user-attachments/assets/f42afa46-8410-42b4-b5af-b5c11b7612ac" />


<img width="448" height="315" alt="image" src="https://github.com/user-attachments/assets/adaca729-c65b-4de1-8c63-c1c67744347e" />


<img width="320" height="204" alt="image" src="https://github.com/user-attachments/assets/1d8d5d34-05d8-44a0-be29-f6242cab70c7" />


<img width="281" height="92" alt="image" src="https://github.com/user-attachments/assets/c8f0553c-49ed-4e78-936e-16b5d9b2edc3" />


<img width="172" height="76" alt="image" src="https://github.com/user-attachments/assets/86dfb012-a358-4da1-92fb-6f40c274d088" 
  />

<img width="318" height="204" alt="image" src="https://github.com/user-attachments/assets/babdef54-b84f-46ff-b457-0accdfeb88f8" />

<img width="1068" height="589" alt="image" src="https://github.com/user-attachments/assets/84b790dc-9561-49a4-bdb6-0edef1a13a4b" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
