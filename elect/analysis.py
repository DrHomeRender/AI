import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# SMAPE 계산 함수
def smape(actual, predicted):
    return 100 * np.mean(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))

# MSE 계산 함수
def mse(actual, predicted):
    return np.mean((predicted - actual) ** 2)

# RMSE 계산 함수
def rmse(actual, predicted):
    return np.sqrt(mse(actual, predicted))

# MAE 계산 함수
def mae(actual, predicted):
    return np.mean(np.abs(predicted - actual))

# R^2 계산 함수
def r2(actual, predicted):
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    return 1 - (ss_res / ss_tot)

# CSV 파일 경로 설정
csv_files = ["./predictions/test_predictions_pool.csv", "./predictions/test_predictions_smart.csv", "./data/train.csv"]
graph_labels = ["data1", "data2", "gnd"]

# 데이터를 저장할 리스트
last_column_data = []

# CSV 파일 읽기 및 마지막 열 데이터 추출
for file in csv_files:
    try:
        data = pd.read_csv(file)
        last_column = data.iloc[:, -1]  # 마지막 열 선택
        last_column_data.append(last_column)
    except Exception as e:
        print(f"Error reading {file}: {e}")
        last_column_data.append(None)

# gnd 데이터 확인
gnd_data = last_column_data[-1]
if gnd_data is None:
    print("Error: gnd data is missing. Cannot calculate scores.")
else:
    scores = {"SMAPE": {}, "MSE": {}, "RMSE": {}, "MAE": {}, "R^2": {}}
    for label, column_data in zip(graph_labels[:-1], last_column_data[:-1]):
        if column_data is not None:
            scores["SMAPE"][label] = smape(gnd_data.values, column_data.values)
            scores["MSE"][label] = mse(gnd_data.values, column_data.values)
            scores["RMSE"][label] = rmse(gnd_data.values, column_data.values)
            scores["MAE"][label] = mae(gnd_data.values, column_data.values)
            scores["R^2"][label] = r2(gnd_data.values, column_data.values)


    # 결과 출력 및 우수한 데이터 판단
    for metric, results in scores.items():
        best_label = min(results, key=results.get) if metric != "R^2" else max(results, key=results.get)
        print(f"{metric} scores:")
        for label, score in results.items():
            print(f"  {label}: {score:.4f}")
        print(f"Best result for {metric}: {best_label} ({results[best_label]:.4f})\n")

# 그래프 그리기
plt.figure(figsize=(20, 12))

for label, column_data in zip(graph_labels, last_column_data):
    if column_data is not None:
        plt.plot(column_data[::20000], label=label, marker='o')  # 데이터 간격을 10으로 설정

plt.title("Comparison of Last Column Data")
plt.xlabel("Index")
plt.ylabel("Values")
plt.legend()
plt.grid(True)

# 그래프 출력
plt.tight_layout()
plt.show()
