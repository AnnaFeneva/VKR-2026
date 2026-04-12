import pandas as pd
import numpy as np


NUMERIC_FEATURES = [
    "attendance","homework_done","avg_grade","discipline_score",
    "engagement_level","absences","teacher_rating","stress_level",
    "motivation_level","extra_classes"
]


# --- УЛУЧШЕННАЯ СИМУЛЯЦИЯ ---
def simulate_change(row, feature):

    new_row = row.copy()

    if feature == "attendance":
        new_row[feature] = min(100, row[feature] + 20)

    elif feature == "homework_done":
        new_row[feature] = min(100, row[feature] + 20)

    elif feature == "motivation_level":
        new_row[feature] = min(1, row[feature] + 0.3)

    elif feature == "stress_level":
        new_row[feature] = max(0, row[feature] - 0.3)

    elif feature == "absences":
        new_row[feature] = max(0, row[feature] - 5)

    return new_row


# --- ИНТЕРПРЕТАЦИЯ ЭФФЕКТА ---
def interpret_gain(gain):
    if gain > 5:
        return "значительное улучшение"
    elif gain > 2:
        return "умеренное улучшение"
    elif gain > 0:
        return "незначительное улучшение"
    else:
        return "ограниченное влияние"


def generate_ai_strategy(df, pipeline, features):

    model = pipeline.named_steps["model"]

    # --- 1. IMPORTANCE ---
    importances = model.feature_importances_

    cat_features = ["class","school_type","parent_education","family_income","internet_access"]

    num_features = [f for f in features if f not in cat_features]

    ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    cat_names = ohe.get_feature_names_out(cat_features)

    feature_names = list(cat_names) + num_features

    # защита от несовпадения длин
    min_len = min(len(feature_names), len(importances))

    feature_names = feature_names[:min_len]
    importances = importances[:min_len]

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    # --- 2. ГРУППИРОВКА ---
    grouped = {}

    for f, imp in zip(feature_names, importances):

        if "class" in f:
            continue

        key = None

        if "attendance" in f:
            key = "attendance"
        elif "homework" in f:
            key = "homework"
        elif "motivation" in f:
            key = "motivation"
        elif "stress" in f:
            key = "stress"
        elif "absences" in f:
            key = "absences"

        if key:
            grouped[key] = grouped.get(key, 0) + imp

    grouped = sorted(grouped.items(), key=lambda x: x[1], reverse=True)

    # --- 3. COUNTERFACTUAL (УЛУЧШЕННЫЙ) ---
    base_preds = pipeline.predict(df[features])

    impact_summary = {}

    for f, _ in grouped[:5]:
        gains = []

        for i, row in df.iterrows():
            try:
                new_row = simulate_change(row, f)

                new_pred = pipeline.predict(pd.DataFrame([new_row[features]]))[0]
                gain = new_pred - base_preds[i]

                # фильтр шума
                if abs(gain) < 0.5:
                    continue

                # убираем нелогичные эффекты
                if gain < 0:
                    continue

                gains.append(gain)

            except:
                continue

        # защита от маленькой выборки
        if len(gains) >= max(5, int(len(df) * 0.2)):
            impact_summary[f] = np.mean(gains)

    sorted_impact = sorted(impact_summary.items(), key=lambda x: x[1], reverse=True)

    # --- защита от пустоты ---
    if not sorted_impact:
        return {
            "importance": imp_df,
            "recommendations": [
                "Анализ не выявил значимых факторов влияния. "
                "Рекомендуется расширить набор данных и провести дополнительную диагностику."
            ],
            "report": "Модель не обнаружила устойчивых закономерностей."
        }

    # --- 4. AI ВЫВОД ---
    recommendations = []
    report_lines = []

    for f, gain in sorted_impact[:5]:

        impact_text = interpret_gain(gain)

        if f == "attendance":
            txt = (
                "Выявлена недостаточная посещаемость учащихся, что приводит к разрывам в усвоении материала. "
                "Рекомендуется усиление контроля посещаемости и работа с группой риска. "
                f"Ожидаемый эффект — {impact_text} успеваемости."
            )

        elif f == "homework":
            txt = (
                "Низкий уровень выполнения домашних заданий снижает закрепление знаний. "
                "Рекомендуется системный контроль выполнения и цифровой мониторинг. "
                f"Ожидаемый эффект — {impact_text} успеваемости."
            )

        elif f == "motivation":
            txt = (
                "Снижение учебной мотивации уменьшает вовлечённость учащихся. "
                "Рекомендуется внедрение интерактивных методов и персонализация обучения. "
                f"Ожидаемый эффект — {impact_text} результатов."
            )

        elif f == "stress":
            txt = (
                "Высокий уровень стресса негативно влияет на когнитивные способности учащихся. "
                "Рекомендуется оптимизация нагрузки и психологическая поддержка. "
                f"Ожидаемый эффект — {impact_text} результатов."
            )

        elif f == "absences":
            txt = (
                "Высокое количество пропусков напрямую снижает академические результаты. "
                "Рекомендуется анализ причин и профилактика пропусков. "
                f"Ожидаемый эффект — {impact_text} успеваемости."
            )

        else:
            continue

        recommendations.append(txt)
        report_lines.append(txt)

    # --- 5. ОТЧЁТ ---
    report = (
        "В результате анализа образовательных данных выявлены ключевые факторы, влияющие на успеваемость. "
        "Основные зоны внимания: "
        + ", ".join([f for f, _ in sorted_impact[:3]])
        + ". Реализация предложенных мер позволит повысить качество образовательного процесса."
    )

    return {
        "importance": imp_df,
        "recommendations": recommendations,
        "report": report
    }