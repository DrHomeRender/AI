import matplotlib.pyplot as plt
import pandas as pd

# 데이터 불러오기
df = pd.read_csv("data_season.csv")

# 1. 열 이름 확인
print("열 이름:", df.columns)

# 2. 연도별 데이터 개수 시각화
year_info = df['Year'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
plt.bar(year_info.index, year_info.values, color='skyblue')
plt.title('Data Count per Year')
plt.xlabel('Year')
plt.ylabel('Count')
plt.grid(True)
plt.show()

# 3. Crops별 평균 가격 시각화
if 'Crops' in df.columns and 'price' in df.columns:
    crops_price = df.groupby('Crops')['price'].mean().sort_values()

    plt.figure(figsize=(14, 8))
    plt.barh(crops_price.index, crops_price.values, color='lightcoral')
    plt.title('Average Price per Crop')
    plt.xlabel('Average Price')
    plt.ylabel('Crops')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()
else:
    print("Crops 또는 price 열이 존재하지 않습니다.")
