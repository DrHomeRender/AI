import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# 데이터 로드
file_path = "data_season.csv"
df = pd.read_csv(file_path)

# 분석할 열
columns_to_analyze = ['yeilds', 'price', 'Rainfall', 'Temperature']

# 그래프 그리기
plt.figure(figsize=(18, 12))

for i, column in enumerate(columns_to_analyze, 1):
    plt.subplot(2, 2, i)
    plot_pacf(df[column], lags=30, method='ywm', ax=plt.gca())
    plt.title(f'Partial Autocorrelation Function (PACF) for {column}')
    plt.xlabel('Lags')
    plt.ylabel('Partial Autocorrelation')
    plt.grid(True)

# 그래프 저장
output_path = "PACF_multiple_plots.png"
plt.tight_layout()
plt.savefig(output_path)
print(f"✅ PACF 그래프가 '{output_path}'로 저장되었습니다.")
