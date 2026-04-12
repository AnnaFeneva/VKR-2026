import pandas as pd
import random

df = pd.read_csv("data/students_realistic_big.csv")

names = ["Иван","Анна","Мария","Алексей","Ольга","Дмитрий","Екатерина"]
surnames = ["Иванов","Петров","Сидоров","Смирнов","Кузнецов"]

df["name"] = [f"{random.choice(names)} {random.choice(surnames)}" for _ in range(len(df))]

df.to_csv("data/students_realistic_big.csv", index=False)
print("Имена добавлены ✅")