import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import pacf

# 데이터 로드
file_path = "data_season.csv"
df = pd.read_csv(file_path)

# 🟢 'time_value'를 시간 기준 열로 설정
time_value = 'Year'  # 연도를 나타내는 컬럼명
df.sort_values(by=time_value, inplace=True)  # 시간 순서 정렬

# 🔢 Lags 자동 계산 (time_value의 고유 연도 수 - 1)
unique_years = df[time_value].unique()  # 고유 연도 값
max_lags = len(unique_years) - 1        # 최대 시차 설정

# 📌 'time_value' 열을 제외한 모든 수치형 항목 찾기
columns_to_analyze = df.select_dtypes(include=['number']).columns.tolist()
columns_to_analyze.remove(time_value)  # 'time_value' 제외

# 분석한 항목 리스트 출력용 문자열 생성
analyzed_columns_str = ", ".join(columns_to_analyze)

# PACF 그래프 그리기
plt.figure(figsize=(18, 12))

for i, column in enumerate(columns_to_analyze, 1):
    plt.subplot((len(columns_to_analyze) + 1) // 2, 2, i)
    plot_pacf(df[column], lags=max_lags, method='ywm', ax=plt.gca())
    plt.title(f'Partial Autocorrelation Function (PACF) for {column}')
    plt.xlabel(f'Lags (Max {max_lags})')
    plt.ylabel('Partial Autocorrelation')
    plt.grid(True)

# 그래프 저장
output_path = "result_PACF.png"
plt.tight_layout()
plt.savefig(output_path)

# 📊 가장 큰 영향을 받는 항목 찾기 (PACF 값 분석)
max_pacf_impact = {}
for column in columns_to_analyze:
    pacf_values = pacf(df[column], nlags=max_lags, method='ywm')
    max_pacf_impact[column] = max(abs(pacf_values[1:]))  # Lag 0 제외

most_affected_feature = max(max_pacf_impact, key=max_pacf_impact.get)

# 🔥 최종 결과 출력
print(f"✅ PACF 그래프가 '{output_path}'로 저장되었습니다.")
print(f"📊 분석한 항목: {analyzed_columns_str}")
print(f"🔍 타임 시리즈 별 가장 큰 영향을 많이 받는 항목은 '{most_affected_feature}' 입니다.")
