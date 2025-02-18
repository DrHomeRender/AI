import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformer import TransformerModel
from data_process import load_data, preprocess_data, split_data

# 모델 파일 경로
MODEL_PATH = "trained_model.pth"
file_path = "data_season.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드 및 전처리
df = load_data(file_path)
df, encoders, scaler, target_scaler = preprocess_data(df)  # target_scaler 추가
X_train, X_test, y_train, y_test = split_data(df)

# PyTorch Tensor 변환
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

# 모델 초기화
input_dim = X_test.shape[1]
output_dim = y_test.shape[1]
model = TransformerModel(input_dim=input_dim, output_dim=output_dim).to(device)

# 저장된 모델 불러오기
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.eval()
print("저장된 모델 로드 완료!")


# 테스트 및 평가
def evaluate():
    criterion = nn.MSELoss()
    with torch.no_grad():
        predictions = model(X_test_tensor.unsqueeze(1))
        loss = criterion(predictions.squeeze(1), y_test_tensor)
        print(f"테스트 MSE Loss: {loss.item():.4f}")

        # 예측값을 원래 스케일로 변환
        predictions_np = predictions.cpu().numpy().squeeze(1)  # 차원 변경 적용
        predictions_original = target_scaler.inverse_transform(predictions_np)
        predictions_original = np.maximum(predictions_original, 0)  # 음수값 방지

        # 예측 결과를 DataFrame으로 정리
        results_df = pd.DataFrame(predictions_original, columns=['Predicted_yeilds', 'Predicted_price'])
        results_df.index.name = "Sample_Index"
        print("예측 결과:")
        print(results_df.head())

    return results_df


if __name__ == "__main__":
    preds_df = evaluate()
    preds_df.to_csv("predictions.csv", index=True)
    print("예측 결과를 'predictions.csv' 파일로 저장 완료!")
