import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split  # ✅ 이 부분 추가

# 파일 경로 및 설정
FILE_PATH = "data_season.csv"
WINDOW_SIZE = 3  # 과거 3년치 데이터를 포함


# 데이터 로드
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# 데이터 전처리 및 변환
def preprocess_data(df, window_size=WINDOW_SIZE):
    # 범주형 변수 인코딩
    categorical_cols = ['Location', 'Soil type', 'Irrigation', 'Crops', 'Season']
    encoders = {col: LabelEncoder() for col in categorical_cols}

    for col in categorical_cols:
        df[col] = encoders[col].fit_transform(df[col])

    # 수치형 데이터 정규화
    numerical_cols = ['Area', 'Rainfall', 'Temperature', 'Humidity', 'yeilds', 'price']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # 슬라이딩 윈도우 적용
    transformed_data = []
    for i in range(window_size, len(df)):
        past_data = df.iloc[i - window_size:i].values.flatten()
        current_data = df.iloc[i].values.flatten()
        transformed_data.append(np.concatenate([past_data, current_data]))

    # 새로운 데이터프레임 생성
    new_columns = []
    for t in range(window_size):
        for col in df.columns:
            new_columns.append(f"Prev{t + 1}_{col}")

    new_columns += [f"Current_{col}" for col in df.columns]
    transformed_df = pd.DataFrame(transformed_data, columns=new_columns)

    # 입력(X)과 타겟(y) 분리
    target_cols = ["Current_yeilds", "Current_price"]
    X = transformed_df.drop(columns=target_cols)
    y = transformed_df[target_cols]

    # 훈련/테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42)  # ✅ train_test_split 사용 가능

    return X_train, X_test, y_train, y_test, encoders, scaler
