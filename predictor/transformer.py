import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, num_layers=6, hidden_dim=512, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # 가중치 초기화
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x  # 활성화 함수 제거


if __name__ == "__main__":
    sample_input = torch.randn(10, 1, 10)
    model = TransformerModel(input_dim=10, output_dim=2)
    output = model(sample_input)
    print("모델 테스트 출력:", output.shape)
