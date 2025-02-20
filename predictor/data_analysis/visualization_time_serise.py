import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 데이터 불러오기
df = pd.read_csv("data_season.csv")
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)

# 모든 작물에 대해 시계열 분해
def analyze_all_crops_seasonality(df, period=3):
    crops = df['Crops'].unique()

    for crop in crops:
        print(f"\n📊 **{crop} 작물 시계열 분석**")
        crop_data = df[df['Crops'] == crop]

        # 연도별 평균 가격 시계열 생성
        avg_price_per_year = crop_data.groupby(crop_data.index.year)['price'].mean()

        # 데이터가 충분하지 않으면 스킵
        if len(avg_price_per_year) < period:
            print(f"⚠️ {crop} 작물은 데이터가 부족하여 분석을 생략합니다.")
            continue

        # 시계열 분해
        try:
            result = seasonal_decompose(avg_price_per_year, model='additive', period=period)
            result.plot()
            plt.suptitle(f'Seasonal Decomposition of {crop} Price')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"❌ {crop} 분석 중 오류 발생: {e}")

# 모든 작물에 대해 시계열 분석 수행
analyze_all_crops_seasonality(df, period=3)
