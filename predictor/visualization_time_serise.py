import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 데이터 불러오기
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Year'] = pd.to_datetime(df['Year'], format='%Y')
    df.set_index('Year', inplace=True)
    return df

# 시계열 시각화 함수
def plot_time_series(df):
    plt.figure(figsize=(14, 10))

    # 수확량 시계열
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['yeilds'], label='Yeilds', color='green')
    plt.title('Yeilds Over Time')
    plt.xlabel('Year')
    plt.ylabel('Yeilds')
    plt.legend()

    # 가격 시계열
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['price'], label='Price', color='orange')
    plt.title('Price Over Time')
    plt.xlabel('Year')
    plt.ylabel('Price')
    plt.legend()

    plt.tight_layout()
    plt.savefig("time_series_plot.png")
    print("📊 시계열 그래프가 'time_series_plot.png'로 저장되었습니다.")

# 시계열 분해 함수
def decompose_time_series(df, column):
    result = seasonal_decompose(df[column], model='additive', period=1)
    result.plot()
    plt.suptitle(f'Seasonal Decomposition of {column}')
    plt.savefig(f"decompose_{column}.png")
    print(f"📊 '{column}' 시계열 분해 그래프가 'decompose_{column}.png'로 저장되었습니다.")

# 메인 실행
if __name__ == "__main__":
    file_path = "data_season.csv"
    df = load_data(file_path)

    # 시계열 시각화
    plot_time_series(df)

    # 시계열 분해
    decompose_time_series(df, 'yeilds')
    decompose_time_series(df, 'price')
