
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = load_wine()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
svm = make_pipeline(StandardScaler(), SVC(kernel="rbf"))

base_models = [
    ("DT", dt),
    ("KNN", knn),
    ("SVM", svm)
]

models = {
    "Decision Tree": dt,
    "KNN": knn,
    "SVM": svm,
    "Stacking": StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5
    ),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
        n_estimators=50,
        random_state=42
    ),
    "Voting": VotingClassifier(
        estimators=base_models,
        voting="hard"
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy (%)": round(accuracy_score(y_test, pred) * 100, 2),
        "Precision (%)": round(precision_score(
            y_test, pred, average="weighted", zero_division=0) * 100, 2),
        "Recall (%)": round(recall_score(
            y_test, pred, average="weighted", zero_division=0) * 100, 2),
        "F1-score (%)": round(f1_score(
            y_test, pred, average="weighted", zero_division=0) * 100, 2)
    })

comparison = pd.DataFrame(results).sort_values(
    "F1-score (%)", ascending=False
)

print("\nMODEL PERFORMANCE COMPARISON")
print(comparison.to_string(index=False))
print("\nBest model:", comparison.iloc[0]["Model"])
