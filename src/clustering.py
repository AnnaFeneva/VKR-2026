from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
import streamlit as st

FEATURES = [
    "attendance","homework_done","avg_grade",
    "engagement_level","absences","stress_level","motivation_level"
]

def cluster_students(df):

    X = df[FEATURES]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    # --- интерпретация кластеров ---
    names = {}

    for c in range(4):
        sub = df[df["cluster"] == c]

        if sub["avg_grade"].mean() > 70:
            names[c] = "Успешные"
        elif sub["stress_level"].mean() > 0.6:
            names[c] = "Перегруженные"
        elif sub["attendance"].mean() < 60:
            names[c] = "Прогульщики"
        else:
            names[c] = "Средние"

    df["student_type"] = df["cluster"].map(names)

    # объяснение кластеров
    def explain_cluster(row):
        if row["student_type"] == "Успешные":
            return "Высокая успеваемость, стабильные показатели"
        elif row["student_type"] == "Перегруженные":
            return "Высокий стресс при нормальной успеваемости"
        elif row["student_type"] == "Прогульщики":
            return "Низкая посещаемость — основной фактор риска"
        else:
            return "Средние показатели по всем метрикам"

    df["cluster_explanation"] = df.apply(explain_cluster, axis=1)

    return df



def show_clusters(df):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="student_type",
        hover_data=["name"]
    )

    st.plotly_chart(fig, use_container_width=True)


def show_cluster_heatmap(df):

    fig = px.imshow(
        df.groupby("student_type")[FEATURES].mean(),
        text_auto=True
    )

    st.plotly_chart(fig, use_container_width=True)


def generate_cluster_insights(df):

    insights = []

    grouped = df.groupby("student_type").mean(numeric_only=True)

    for name, row in grouped.iterrows():

        text = f"Группа '{name}': "

        if row["avg_grade"] < 3:
            text += "низкая успеваемость; "

        if row["attendance"] < 60:
            text += "проблемы с посещаемостью; "

        if row["stress_level"] > 0.6:
            text += "высокий уровень стресса; "

        if row["motivation_level"] < 0.5:
            text += "низкая мотивация; "

        insights.append(text)

    return insights


def generate_class_recommendations(df):

    recs = []

    if df["attendance"].mean() < 65:
        recs.append("Усилить контроль посещаемости")

    if df["stress_level"].mean() > 0.6:
        recs.append("Снизить учебную нагрузку")

    if df["motivation_level"].mean() < 0.5:
        recs.append("Повысить вовлеченность учащихся")

    if not recs:
        recs.append("Система работает стабильно")

    return recs
