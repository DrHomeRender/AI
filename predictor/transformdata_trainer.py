import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from transformer import TransformerModel
from data_transform import load_data, preprocess_data  # data_process 대신 data_transform 사용

# 하이퍼파라미터 설정
BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 0.001
MODEL_BEST_PATH = "best_trans_data_model.pth"
MODEL_LATEST_PATH = "latest_trans_data_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 데이터 로드 및 전처리
file_path = "data_season.csv"
df = load_data(file_path)
X_train, X_test, y_train, y_test, encoders, scaler = preprocess_data(df)  # data_transform.py 사용

# PyTorch Tensor 변환
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# 데이터 로더 생성
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 모델 초기화
input_dim = X_train.shape[1]  # 슬라이딩 윈도우를 적용했으므로 input_dim 증가
output_dim = y_train.shape[1]  # 수확량과 가격 예측 (2개 출력)
model = TransformerModel(input_dim=input_dim, output_dim=output_dim).to(device)

# 기존 모델이 존재하면 불러오기
best_loss = float("inf")
if os.path.exists(MODEL_LATEST_PATH):
    model.load_state_dict(torch.load(MODEL_LATEST_PATH, map_location=device))
    print("기존 최신 모델 로드 완료! 추가 학습 진행합니다.")
else:
    print("새로운 모델 생성!")

# 손실 함수 및 옵티마이저 설정
criterion = nn.SmoothL1Loss(beta=1.0)  # MSE 대신 이상치에 강한 SmoothL1Loss 사용
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)  # AdamW 적용
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# 학습 과정 시각화를 위한 변수
losses = []

# 학습 루프
def train():
    global best_loss
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x.unsqueeze(1))  # Transformer는 (Batch, Sequence, Feature) 형태 필요

            loss = criterion(outputs.squeeze(1), batch_y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient Clipping 추가
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)  # Loss 저장
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.4f}")

        # Best 모델 저장 (최저 Loss 갱신 시)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_BEST_PATH)
            print(f"✅ 새로운 Best 모델 저장 (Loss: {best_loss:.4f})")

        # 최신 모델 저장
        torch.save(model.state_dict(), MODEL_LATEST_PATH)

        # 학습률 스케줄러 업데이트
        scheduler.step(avg_loss)

    print("모델 저장 완료!")

    # Loss 그래프 저장
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Train Loss", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    plt.grid()
    plt.savefig("loss_curve.png")
    print("📊 학습 Loss 그래프 저장 완료 (loss_curve.png)")


if __name__ == "__main__":
    train()
