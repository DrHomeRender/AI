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
def plot_time_series(df, columns):
    plt.figure(figsize=(14, 6 * len(columns)))

    for i, column in enumerate(columns, 1):
        if column in df.columns:
            plt.subplot(len(columns), 1, i)
            plt.plot(df.index, df[column], label=column, color='skyblue')
            plt.title(f'{column} Over Time')
            plt.xlabel('Year')
            plt.ylabel(column)
            plt.legend()
        else:
            print(f"⚠️ '{column}' 열이 존재하지 않습니다.")

    plt.tight_layout()
    plt.savefig("time_series_plot.png")
    print("📊 시계열 그래프가 'time_series_plot.png'로 저장되었습니다.")
    plt.show()

# 시계열 분해 함수
def decompose_time_series(df, column, period=5):
    if column not in df.columns:
        print(f"⚠️ '{column}' 열이 존재하지 않습니다.")
        return

    try:
        result = seasonal_decompose(df[column], model='additive', period=period)
        result.plot()
        plt.suptitle(f'Seasonal Decomposition of {column}')
        plt.savefig(f"decompose_{column}.png")
        print(f"📊 '{column}' 시계열 분해 그래프가 'decompose_{column}.png'로 저장되었습니다.")
        plt.show()
    except Exception as e:
        print(f"❌ 시계열 분해 중 오류 발생: {e}")

# 메인 실행
if __name__ == "__main__":
    file_path = "data_season.csv"
    df = load_data(file_path)
    print(df.head())

    # 시계열 시각화 (분석할 열 선택)
    target_columns = ['yeilds', 'price']
    plot_time_series(df, target_columns)

    # 시계열 분해 (주기 설정)
    for column in target_columns:
        decompose_time_series(df, column, period=3)
