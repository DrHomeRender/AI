import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from transformer import TransformerModel
from data_transform import load_data, preprocess_data  # ✅ 데이터 변환 코드 사용

print("GPU 사용 가능 여부:", torch.cuda.is_available())
print("사용 중인 GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
print("현재 사용 중인 디바이스:", torch.cuda.current_device())

# ✅ 하이퍼파라미터 설정
BATCH_SIZE = 32
EPOCHS = 1000
LEARNING_RATE = 0.001
MODEL_BEST_PATH = "best.pth"
MODEL_LATEST_PATH = "latest.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 데이터 로드 및 전처리
file_path = "data_season.csv"
df = load_data(file_path)
X_train, X_test, y_train, y_test, encoders, scaler = preprocess_data(df)

# ✅ PyTorch Tensor 변환
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# ✅ 데이터 로더 생성 (최적화)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=os.cpu_count()
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=os.cpu_count()
)

# ✅ 모델 초기화
input_dim = X_train.shape[1]  # 슬라이딩 윈도우 적용으로 input_dim 증가
output_dim = y_train.shape[1]  # 수확량과 가격 예측 (2개 출력)
model = TransformerModel(input_dim=input_dim, output_dim=output_dim).to(device)

# ✅ 기존 모델 불러오기
best_loss = float("inf")
if os.path.exists(MODEL_LATEST_PATH):
    model.load_state_dict(torch.load(MODEL_LATEST_PATH, map_location=device))
    print("🔄 기존 최신 모델 로드 완료! 추가 학습 진행합니다.")
else:
    print("🆕 새로운 모델 생성!")

# ✅ 손실 함수 및 옵티마이저 설정
criterion = nn.MSELoss()  # ✅ MSELoss 사용 (SmoothL1Loss도 가능)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)  # ✅ AdamW 적용
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)  # ✅ 동적 학습률 조절

# ✅ 학습 과정 시각화를 위한 변수
losses = []

# ✅ 학습 루프
def train():
    global best_loss
    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(batch_x.unsqueeze(1))  # ✅ Transformer 입력 차원 맞추기
            loss = criterion(outputs.squeeze(1), batch_y)
            loss.backward()
            
            # ✅ Gradient Clipping 적용 (기울기 폭발 방지)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)  # ✅ Loss 저장
        print(f"📝 Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")

        # ✅ Best 모델 저장 (최저 Loss 갱신 시)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), MODEL_BEST_PATH)
            print(f"✅ 새로운 Best 모델 저장 (Loss: {best_loss:.6f})")

        # ✅ 최신 모델 저장 (10 Epoch마다)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), MODEL_LATEST_PATH)

        # ✅ 학습률 스케줄러 업데이트 (Loss 기반 감소)
        scheduler.step(avg_loss)

    print("✅ 모델 학습 및 저장 완료!")

    # ✅ Loss 그래프 저장
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
