import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
class DataPreparation:
    def __init__(self, base_path):
        self.base_path = base_path
    #요일별 전력 소비 패턴의 차이를 반영하기 위해, target 값을 가중치(ratio)를 통해 조정.
    #예: 월요일과 일요일의 전력 소비 패턴이 다를 수 있으므로, 요일별로 target 값을 조정하여 학습 데이터를 정제.
    # 특징 엔지니어링
    #요일별 가중치로 target 조정.
    #통계적 평균 및 표준 편차 특징 생성.
    def meanStd(self, train, test):
        # 요일별(dow) 전력 소비량 조정
        ratio = np.array([0.98, 0.98, 0.995, 0.995, 0.99, 0.99, 0.97])

        train['date_time'] = train['num_date_time'].str.split('_').str[1]
        test['date_time'] = test['num_date_time'].str.split('_').str[1]
        train['date_time'] = pd.to_datetime(train['date_time'], format='%Y%m%d %H', errors='coerce')
        test['date_time'] = pd.to_datetime(test['date_time'], format='%Y%m%d %H', errors='coerce')

        train['hour'] = train['date_time'].dt.hour
        test['hour'] = test['date_time'].dt.hour

        train['dow'] = train['date_time'].dt.weekday
        test['dow'] = test['date_time'].dt.weekday

        train['holiday'] = train['dow'].apply(lambda x: 1 if x >= 5 else 0)
        test['holiday'] = test['dow'].apply(lambda x: 1 if x >= 5 else 0)

        train['target'] = train.apply(lambda row: row['target'] * ratio[row['dow']], axis=1)

        # 평균 및 표준편차 계산
        power_mean = pd.pivot_table(train, values='target', index=['building', 'hour', 'dow'],
                                    aggfunc=np.mean).reset_index()
        power_mean.rename(columns={'target': 'dow_hour_mean'}, inplace=True)
        train = pd.merge(train, power_mean, on=['building', 'hour', 'dow'], how='left')
        test = pd.merge(test, power_mean, on=['building', 'hour', 'dow'], how='left')

        power_holiday_mean = pd.pivot_table(train, values='target', index=['building', 'hour', 'holiday'],
                                            aggfunc=np.mean).reset_index()
        power_holiday_mean.rename(columns={'target': 'holiday_mean'}, inplace=True)
        train = pd.merge(train, power_holiday_mean, on=['building', 'hour', 'holiday'], how='left')
        test = pd.merge(test, power_holiday_mean, on=['building', 'hour', 'holiday'], how='left')

        power_holiday_std = pd.pivot_table(train, values='target', index=['building', 'hour', 'holiday'],
                                           aggfunc=np.std).reset_index()
        power_holiday_std.rename(columns={'target': 'holiday_std'}, inplace=True)
        train = pd.merge(train, power_holiday_std, on=['building', 'hour', 'holiday'], how='left')
        test = pd.merge(test, power_holiday_std, on=['building', 'hour', 'holiday'], how='left')

        power_hour_mean = pd.pivot_table(train, values='target', index=['building', 'hour'],
                                         aggfunc=np.mean).reset_index()
        power_hour_mean.rename(columns={'target': 'hour_mean'}, inplace=True)
        train = pd.merge(train, power_hour_mean, on=['building', 'hour'], how='left')
        test = pd.merge(test, power_hour_mean, on=['building', 'hour'], how='left')

        power_hour_std = pd.pivot_table(train, values='target', index=['building', 'hour'],
                                        aggfunc=np.std).reset_index()
        power_hour_std.rename(columns={'target': 'hour_std'}, inplace=True)
        train = pd.merge(train, power_hour_std, on=['building', 'hour'], how='left')
        test = pd.merge(test, power_hour_std, on=['building', 'hour'], how='left')

        return train, test

    # 데이터 불러오기 및 컬럼 값 영어로 바꿈, 일조 일사 테스트에 없기 때문에 드랍
    def load_data(self):
        """
        Load building_info.csv, train.csv, and test.csv files from the base path.
        """
        building_info = pd.read_csv(os.path.join(self.base_path, 'building_info.csv')).drop(['ESS저장용량(kWh)', 'PCS용량(kW)'], axis=1)
        train_data = pd.read_csv(os.path.join(self.base_path, 'train.csv')).drop(['일조(hr)', '일사(MJ/m2)'], axis=1)
        test_data = pd.read_csv(os.path.join(self.base_path, 'test.csv'))

        # 한글 colums를 영어로 변환 [num_date_time, 건물번호, 일시, 기온, 강수량, 풍속, 습도, 일조, 일사, 전력소비량]
        train_data.columns = ['num_date_time', 'building', 'date_time', 'temp', 'prec', 'wind', 'hum', 'target']
        test_data.columns = ['num_date_time', 'building', 'date_time', 'temp', 'prec', 'wind', 'hum']

        return building_info, train_data, test_data

    def process_building_info(self, building_info):
        """
        Process the building_info.csv file to handle missing values and normalize data.
        """
        # 칼럼값 영어로 바꿈 [건물 번호, 건물유형, 연면적, 냉방면적, 태양광 용량]
        building_info.columns = ['building', 'type', 'all_area', 'cool_area', 'sun']
        # None 값으로 "-" 로 표시했었는데 다 0으로 바꿈 "태양광 용량"
        building_info['sun'] = building_info['sun'].replace('-', 0).astype('float')

        # 라벨링 인코딩을 위한 처리
        # 문자열 인덱스에 딕션너리로 숫자값을 부여 예) value_dict = {'Office': 0, 'Residential': 1, 'Commercial': 2}
        value_dict = {value: index for index, value in enumerate(building_info['type'].unique())}
        # 타입값에. 숫자값을 매핑
        building_info['type'] = building_info['type'].map(value_dict)

        # 건물 유형이 7인(아파트) 데이터만 선택 냉방면적이 0이 아닌 데이터만 선택
        filtered_data = building_info[(building_info['type'] == 7) & (building_info['cool_area'] != 0)]
        # iloc[1:] 첫번째 행을 제외하고 나머지 행만 선택 sum 총면적과 냉방면적의 합계 계산 이후 총면적/ 냉방면적의 비율
        # 냉방면적이 0으로 설정된 경우 처리하기 위한 평균 비율을 구함
        result = (filtered_data['all_area'].iloc[1:].sum() / filtered_data['cool_area'].iloc[1:].sum())
        # 건물이 아파트고 냉방면적 0 인것을 선택
        condition = (building_info['type'] == 7) & (building_info['cool_area'] == 0)
        # 조건을 만족하는 행의  냉방면적 열을 선택, = all area를 앞에서 계산한 result로 나눔 계산결과를 정수로 변환
        # 냉방면적이 0 인경우 총면적을 기준으로 냉방면적을 계산하여 채움
        building_info.loc[condition, 'cool_area'] = (building_info.loc[condition, 'all_area'] / result).astype('int')

        # 1. 건물 유형(type)이 9(지식산업센터)이고 냉방 면적(cool_area)이 500 이상인 데이터를 필터링
        filtered_data = building_info[(building_info['type'] == 9) & (building_info['cool_area'] > 500)]

        # 2. 필터링된 데이터에서 총 면적(all_area)과 냉방 면적(cool_area)의 비율 계산
        #    - result는 냉방 면적 대비 총 면적의 평균 비율을 나타냄
        result = (filtered_data['all_area'].sum() / filtered_data['cool_area'].sum())

        # 3. 건물 유형(type)이 9이고 냉방 면적(cool_area)이 500 미만인 데이터를 선택하는 조건 생성
        condition = (building_info['type'] == 9) & (building_info['cool_area'] < 500)

        # 4. 위 조건을 만족하는 데이터에서 냉방 면적(cool_area)을 비율(result)을 이용해 계산하여 갱신
        #    - 냉방 면적 = 총 면적 / 비율(result)
        #    - 계산된 값을 소수점 첫째 자리로 반올림하여 cool_area 열에 할당
        building_info.loc[condition, 'cool_area'] = round(building_info.loc[condition, 'all_area'] / result, 1)

        return building_info
    #특징 엔지니어링
    #시간 관련 특징 생성 (hour, dow, sin_time 등).
    #결측값 처리 및 공휴일 플래그 추가.
    def process_data(self, data):
        """
        Preprocess train or test data: handle missing values, create time features, and normalize data.
        """
        # 기본형 DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None)
        # 추천 개선 방법
        # 첫 값이 결측일 경우 **백필링(backfill)**이나 다른 통계적 추정 방법 사용:
        # data['wind'] = data['wind'].fillna(method='ffill').fillna(method='bfill')
        # 특정 열에 대해 평균이나 중앙값으로 결측값을 채우는 방식도 고려:
        # data['wind'] = data['wind'].fillna(data['wind'].mean())
        # wind와 hum 열의 결측값을 각각 이전 값으로 채움
        # 풍속(wind)과 습도(hum)는 시간에 따라 연속적인 특성을 가지므로, 앞의 값으로 결측값을 채우는 것이 데이터의 연속성을 유지
        data['wind'] = data['wind'].fillna(method='ffill')
        data['hum'] = data['hum'].fillna(method='ffill')
        data = data.fillna(0) # 나머지는 0으로 채움

        data['date_time'] = data['num_date_time'].str.split('_').str[1]  # Extract datetime part
        data['date'] = pd.to_datetime(data['date_time'], format='%Y%m%d %H', errors='coerce')

        # Time-related features
        data['hour'] = data['date'].dt.hour
        data['dow'] = data['date'].dt.weekday
        data['month'] = data['date'].dt.month
        # date: 2022-08-25 00:00:00
        # week: 34 ISO-8601 기준으로 주 번호를 계산하여 week 열에 저장.
        # 머신러닝 모델에서 시간 데이터는 중요한 영향을 미치는 경우가 많음
        data['week'] = data['date'].dt.isocalendar().week.astype(np.int32)
        data['day'] = data['date'].dt.day

        #하루 24시간을 기준으로, 시간 데이터를 0–1 범위의 사인 값으로 변환.
        '''
            hour: 0, sin_time: 0.0
            hour: 6, sin_time: 1.0
            hour: 12, sin_time: 0.0
            hour: 18, sin_time: -1.0
        '''
        # 머신러닝 모델에서 시간 데이터를 단순 숫자로 사용할 경우, 주기적인 특성이 제대로 학습되지 않을 수 있음.
        # sin_time 값은 시간에 따른 주기적인 변화를 표현
        # cos 값은 sin과 직교하는 값으로, 두 값을 함께 사용하면 2D 공간에서 특정 시간대를 정확히 표현

        data['sin_time'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['cos_time'] = np.cos(2 * np.pi * data['hour'] / 24)

        # 요일(dow)을 기반으로 공휴일 여부를 나타내는 이진 플래그(1/0)를 생성
        # 공휴일은 많은 데이터(예: 에너지 소비량, 교통량 등)에 영향을 미침.
        # dow 값이 5(토요일) 또는 6(일요일)일 경우, 공휴일로 간주하고 1로 설정.

        data['holiday'] = data['dow'].apply(lambda x: 1 if x >= 5 else 0)

        return data

    def prepare(self):
        """
        Main method to load, preprocess, and return all necessary datasets.
        """
        building_info, train_data, test_data = self.load_data()
        building_info = self.process_building_info(building_info)
        train_data, test_data = self.meanStd(train_data, test_data)

        # Step 5: Merge building_info with train and test data
        train_data = train_data.merge(building_info, on='building', how='left')
        test_data = test_data.merge(building_info, on='building', how='left')

        return train_data, test_data

# Example usage
if __name__ == "__main__":
    base_path = './data'
    data_preparation = DataPreparation(base_path)
    train_data, test_data = data_preparation.prepare()

    # 시간별 소비량 평균
    hourly_mean = train_data.groupby('hour')['target'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_mean.index, hourly_mean.values, marker='o', label='Hourly Mean Consumption')
    plt.title('Hourly Mean Power Consumption')
    plt.xlabel('Hour')
    plt.ylabel('Mean Consumption')
    plt.grid()
    plt.legend()
    plt.show()

    # 공휴일 vs 평일 소비량 비교
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=train_data, x='holiday', y='target')
    plt.title('Power Consumption: Holiday vs Weekday')
    plt.xlabel('Holiday (1: Yes, 0: No)')
    plt.ylabel('Power Consumption')
    plt.show()

    # 건물별 소비량 비교
    building_mean = train_data.groupby('building')['target'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 8))
    building_mean.plot(kind='bar', color='skyblue')
    plt.title('Average Power Consumption by Building')
    plt.xlabel('Building')
    plt.ylabel('Average Power Consumption')
    plt.show()