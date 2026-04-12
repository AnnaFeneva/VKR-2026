import joblib

from src.ai_recommender import generate_ai_recommendations
from src.risk_model import train_risk_model, calculate_risk
from src.clustering import cluster_students


REQUIRED_COLUMNS = [
    "class","school_type","parent_education","family_income","internet_access",
    "attendance","homework_done","avg_grade","discipline_score",
    "engagement_level","absences","teacher_rating","stress_level",
    "motivation_level","extra_classes"
]


def run_pipeline(df):

    pipeline = joblib.load("models/pipeline.pkl")

    X = df[REQUIRED_COLUMNS].copy()

    # --- 1. прогноз успеваемости ---
    df["predicted_score"] = pipeline.predict(X)

    # --- 2. AI рекомендации ---
    df = generate_ai_recommendations(df)

    # --- 3. обучение risk модели ---
    risk_model, scaler = train_risk_model(df, REQUIRED_COLUMNS)

    # --- 4. расчет риска ---
    df = calculate_risk(df, risk_model, scaler, REQUIRED_COLUMNS)

    # --- 5. кластеризация ---
    df = cluster_students(df)

    return df, pipeline, REQUIRED_COLUMNS