
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = VotingClassifier(
    estimators=[
        ("DT", DecisionTreeClassifier(max_depth=5, random_state=42)),
        ("KNN", make_pipeline(StandardScaler(),
                              KNeighborsClassifier(n_neighbors=5))),
        ("SVM", make_pipeline(StandardScaler(), SVC(kernel="rbf")))
    ],
    voting="hard"
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Voting Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=data.target_names))
