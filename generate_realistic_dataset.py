import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)
N = 5000
np.random.seed(42)

df = pd.DataFrame({
    "student_id": np.arange(N),
    "class": np.random.choice(["9A","9B","10A","10B","11A"], N),
    "school_type": np.random.choice(["urban","rural"], N),
    "attendance": np.random.uniform(50,100,N),
    "homework_done": np.random.randint(0,100,N),
    "exam_score": np.random.randint(0,100,N),
    "avg_grade": np.random.uniform(2,5,N),
    "discipline_score": np.random.uniform(0,1,N),
    "engagement_level": np.random.uniform(0,1,N),
    "absences": np.random.randint(0,30,N),
    "parent_education": np.random.choice(["low","medium","high"], N),
    "family_income": np.random.choice(["low","middle","high"], N),
    "internet_access": np.random.choice([0,1], N),
    "teacher_rating": np.random.uniform(0,1,N),
    "stress_level": np.random.uniform(0,1,N),
    "motivation_level": np.random.uniform(0,1,N),
    "extra_classes": np.random.choice([0,1], N),
})

df["exam_score"] = (
    df["exam_score"] * 0.5 +
    df["attendance"] * 0.2 +
    df["homework_done"] * 0.2 +
    df["motivation_level"] * 100 * 0.1
)
df["exam_score"] = df["exam_score"].clip(0,100)
df.to_csv("data/students_realistic_big.csv", index=False)
print("Dataset created:", df.shape)