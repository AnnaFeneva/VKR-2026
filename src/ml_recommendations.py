import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

def ml_recommendations(df):
    df_copy = df.copy()
    
    features = [
        "attendance",
        "homework_done",
        "discipline_score",
        "engagement_level",
        "absences",
        "internet_access",
        "teacher_rating",
        "stress_level",
        "motivation_level",
        "extra_classes"
    ]
    
    # Преобразуем в числа
    for col in features:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').fillna(0)
    
    X = df_copy[features]
    y = df_copy["exam_score"]

    # Pipeline с scaler и моделью
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(n_estimators=50, random_state=42))
    ])

    pipeline.fit(X, y)
    
    df_copy["predicted_score"] = pipeline.predict(X)
    
    return df_copy, pipeline, features