import pandas as pd
import joblib
import os

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/students_realistic_big.csv")

target = "exam_score"

cat_features = ["class","school_type","parent_education","family_income","internet_access"]

num_features = [
    "attendance","homework_done","avg_grade","discipline_score",
    "engagement_level","absences","teacher_rating","stress_level",
    "motivation_level","extra_classes"
]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
], remainder="passthrough")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ))
])

X = df[cat_features + num_features]
y = df[target]

pipeline.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/pipeline.pkl")

print("Модель обучена ✅")