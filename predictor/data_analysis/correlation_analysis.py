import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 데이터 불러오기
df = pd.read_csv("data_season.csv")


# 범주형 열과 수치형 열 구분
def separate_columns(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    return categorical_cols, numeric_cols


# 범주형 열을 레이블 인코딩
def encode_categorical(df):
    categorical_cols, _ = separate_columns(df)
    label_encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = label_encoder.fit_transform(df[col])

    return df


# 상관관계 분석 함수 (mode 기능 추가)
def correlation_analysis(df, mode="all"):
    corr_matrix = df.corr(numeric_only=True)

    # 1. 전체 상관관계 분석
    if mode == "all":
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
        plt.title('Correlation Heatmap (All Variables)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    # 2. 특정 열 중심 상관관계 분석
    elif mode in df.columns:
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix[[mode]].sort_values(by=mode, ascending=False),
                    annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
        plt.title(f'Correlation Heatmap with "{mode}"')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    else:
        print(f"Error: '{mode}' is not a valid column name.")


if __name__ == '__main__':
    # 범주형 열을 레이블 인코딩
    df_encoded = encode_categorical(df)

    # 전체 상관관계 분석
    correlation_analysis(df_encoded)

    # 특정 열과의 상관관계 분석 (예: price 기준)
    correlation_analysis(df_encoded, mode="price")
