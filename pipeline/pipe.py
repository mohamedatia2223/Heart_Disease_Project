import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("heart_disease_uci.csv")
df = df.drop(columns=['id','dataset'], errors="ignore")

numCol = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
catCol = ["sex", "cp", "restecg", "slope", "thal"]
boolCol = ["fbs", "exang"]

df['num'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

X = df.drop('num', axis=1)
y = df['num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

boolean_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numCol),
        ("cat", categorical_transformer, catCol),
        ("bool", boolean_transformer, boolCol)
    ]
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("feature_selection", SelectFromModel(RandomForestClassifier(random_state=42))),
    ("pca", PCA(n_components=8)),
    ("classifier", SVC())
])

param_grid_svm = [
    {
        "classifier__kernel": ["linear"],
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__max_iter": [1000, -1]
    },
    {
        "classifier__kernel": ["rbf"],
        "classifier__C": [0.1, 1, 10],
        "classifier__gamma": ["scale", "auto", 0.01, 0.1, 1],
        "classifier__max_iter": [1000, -1]
    }
]

grid_svm = GridSearchCV(
    pipeline, param_grid=param_grid_svm, cv=5, verbose=1
)

grid_svm.fit(X_train, y_train)

best_model = grid_svm.best_estimator_

y_pred = best_model.predict(X_test)
print("Best Parameters:", grid_svm.best_params_)
print("Best CV Score:", grid_svm.best_score_)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix (SVM)")
plt.show()

joblib.dump(best_model, "heart_disease_pipeline_svm.pkl")
print("✅ Full pipeline saved as heart_disease_pipeline_svm.pkl")
