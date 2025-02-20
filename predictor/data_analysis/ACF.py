import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

# 데이터 로드
file_path = "data_season.csv"
df = pd.read_csv(file_path)

# 🟢 'time_value'를 시간 기준 열로 설정
time_value = 'Year'  # 연도를 나타내는 컬럼명
df.sort_values(by=time_value, inplace=True)

# 🔢 Lags 자동 계산 (time_value의 고유 연도 수 - 1)
unique_years = df[time_value].unique()
max_lags = len(unique_years) - 1

# 📌 'time_value' 열을 제외한 모든 수치형 항목 찾기
columns_to_analyze = df.select_dtypes(include=['number']).columns.tolist()
columns_to_analyze.remove(time_value)

# 분석한 항목 리스트 출력용 문자열 생성
analyzed_columns_str = ", ".join(columns_to_analyze)

# ACF 그래프 그리기
plt.figure(figsize=(18, 12))

for i, column in enumerate(columns_to_analyze, 1):
    plt.subplot((len(columns_to_analyze) + 1) // 2, 2, i)
    plot_acf(df[column], lags=max_lags, ax=plt.gca())
    plt.title(f'Autocorrelation Function (ACF) for {column}')
    plt.xlabel(f'Lags (Max {max_lags})')
    plt.ylabel('Autocorrelation')
    plt.grid(True)

# 그래프 저장
output_path = "result_ACF.png"
plt.tight_layout()
plt.savefig(output_path)

# 📊 가장 큰 영향을 받는 항목 찾기 (ACF 값 분석)
max_acf_impact = {}
for column in columns_to_analyze:
    acf_values = acf(df[column], nlags=max_lags)
    max_acf_impact[column] = max(abs(acf_values[1:]))  # Lag 0 제외

most_affected_feature = max(max_acf_impact, key=max_acf_impact.get)

# 🔥 최종 결과 출력
print(f"✅ ACF 그래프가 '{output_path}'로 저장되었습니다.")
print(f"📊 분석한 항목: {analyzed_columns_str}")
print(f"🔍 타임 시리즈 별 가장 큰 영향을 많이 받는 항목은 '{most_affected_feature}' 입니다.")
