import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data_season.csv")
# 처음 데이터 탐색
print("head")
print(df.head())
print("첫행")
print("한 행에 나와있는 모든 열의 내용")
print(df.loc[0])
print(len(df.loc[0]))
print("데이터의 총량")
print(len(df))
print("데이터 정보")
print(df.info)


# 데이터 선택해서 보기 조건 필터링
print(df[df['Year']>2017])

print(df[df['Year']==2017])

# 각속성 값이 무엇이 있는지 알고 싶을때
print("연도가 어떤 값들이 있는지 알아보기",df['Year'].unique())
print(sorted(df['Year'].unique()))
print(df["Crops"].unique())

# 각 속성 값이 무엇이있고 얼마나 있는지 알고 싶을때
year_info = df['Year'].value_counts().sort_index()
print(year_info)

plt.figure(figsize=(12, 6))
plt.bar(year_info.index, year_info.values, color='skyblue')
plt.xlabel('Year')
plt.ylabel('Count')
plt.title('Number of Records per Year')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()