import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# 저장 경로 설정
output_dir = "./"
output_path = os.path.join(output_dir, "sin_data.csv")

#  사인 데이터 생성
time = np.arange(0, 100, 0.1)
value = np.sin(time)

# DataFrame으로 변환 후 저장
df = pd.DataFrame({"time": time, "value": value})
df.to_csv(output_path, index=False)

#  원본 데이터를 바로 시각화
plt.figure(figsize=(12, 6))
plt.plot(time, value, label="sin_data", color="blue")
plt.title("Generated Sine Wave (Original Data)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#  CSV로부터 불러온 데이터를 시각화
csv_path = "sin_data.csv"
df_loaded = pd.read_csv(csv_path)

plt.figure(figsize=(10, 4))
plt.plot(df_loaded["time"], df_loaded["value"], label="sin(t)", color='green')
plt.title("Loaded Sine Wave from CSV")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
