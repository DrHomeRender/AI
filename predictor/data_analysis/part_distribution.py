import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("data_season.csv")
# 열 이름만 확인
print(df.columns)

# 연도별 데이터 개수 시각화
plt.figure(figsize=(12, 6))
df['Soil type'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('data_number')
plt.xlabel('Year')
plt.ylabel('Count')
plt.grid(True)
plt.show()
