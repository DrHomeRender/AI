import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# 데이터 불러오기
df = pd.read_csv("data_season.csv")

# 분석할 열 목록
columns_to_analyze = ['yeilds', 'price', 'Rainfall', 'Temperature']

# ACF 플롯 그리기 및 이미지로 저장
plt.figure(figsize=(12, 10))

for i, column in enumerate(columns_to_analyze, 1):
    plt.subplot(2, 2, i)
    plot_acf(df[column], lags=20, ax=plt.gca())
    plt.title(f"ACF Plot for {column}")
    plt.grid()

plt.tight_layout()
plt.savefig("acf_multiple_plots.png")
plt.close()

print("✅ 여러 열에 대한 ACF 그래프가 'acf_multiple_plots.png'로 저장되었습니다.")
