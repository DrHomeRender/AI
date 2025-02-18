import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Transformer를 위한 포지셔널 인코딩"""
    def __init__(self, hidden_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, hidden_dim)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, num_layers=4, hidden_dim=384, dropout=0.1, seq_len=1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)
        self.seq_len = seq_len

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
        x = self.pos_encoder(x)  # 포지셔널 인코딩 추가
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x

if __name__ == "__main__":
    sample_input = torch.randn(10, 1, 10)  # (batch_size, seq_len, input_dim)
    model = TransformerModel(input_dim=10, output_dim=2)
    output = model(sample_input)
    print("모델 테스트 출력:", output.shape)
