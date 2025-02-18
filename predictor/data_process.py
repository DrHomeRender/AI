import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path):
    """ 데이터 로드 """
    df = pd.read_csv(file_path)
    print("데이터 로드 완료. 샘플 5개:")
    print(df.head())
    print("데이터 정보:")
    print(df.info())
    return df


def preprocess_data(df):
    """ 데이터 전처리 """
    print("전처리 시작...")

    # 데이터 복사하여 원본 유지
    df = df.copy()

    # 결측값 처리 (Soil type)
    df['Soil type'] = df['Soil type'].fillna('Unknown')  # 'Unknown'으로 채움

    # 결측값 확인
    print("결측값 개수:")
    print(df.isnull().sum())

    # 연도 정규화 (최소 연도를 기준으로 변경)
    df['Year'] = df['Year'] - df['Year'].min()

    # 범주형 데이터 인코딩
    categorical_cols = ['Location', 'Soil type', 'Irrigation', 'Crops', 'Season']
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"{col} 인코딩 완료. 예시: {df[col].unique()[:5]}")

    # 수치형 데이터 정규화
    numerical_cols = ['Area', 'Rainfall', 'Temperature', 'Humidity']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    print("수치형 데이터 정규화 완료.")

    # 출력값 정규화 (yeilds, price)
    target_cols = ['yeilds', 'price']
    target_scaler = StandardScaler()
    df[target_cols] = target_scaler.fit_transform(df[target_cols])
    print("출력값 정규화 완료.")

    print("전처리 완료. 샘플 5개:")
    print(df.head())
    return df, label_encoders, scaler, target_scaler


def split_data(df):
    """ 훈련/테스트 데이터 분할 """
    X = df.drop(columns=['yeilds', 'price'])  # price(가격)도 예측 목표로 추가
    y = df[['yeilds', 'price']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y['yeilds'])
    print("훈련 데이터 크기:", X_train.shape, "테스트 데이터 크기:", X_test.shape)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # 테스트 실행
    file_path = "data_season.csv"  # 실제 데이터 파일 경로로 변경
    df = load_data(file_path)
    df, encoders, scaler, target_scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(df)

    print("최종 데이터셋 샘플:")
    print("훈련 데이터 X 샘플:")
    print(X_train.head())
    print("훈련 데이터 y 샘플:")
    print(y_train.head())
