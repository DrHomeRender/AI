import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pe = None  # Positional Encoding을 미리 생성하지 않고, 필요할 때 생성

    def forward(self, x):
        seq_len = x.size(1)

        # 새로운 Sequence Length에 맞춰 Positional Encoding 생성
        if self.pe is None or self.pe.shape[1] < seq_len:
            pe = torch.zeros(1, seq_len, self.d_model)
            position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.d_model))
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.pe = pe.to(x.device)  # Positional Encoding을 동적으로 생성

        return x + self.pe[:, :seq_len, :]
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=8, num_layers=6, hidden_dim=512, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)  # ✅ Positional Encoding 추가

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

        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        #print(f"🔍 입력 데이터 크기: {x.shape}")  # 디버깅용
        x = self.embedding(x)
        x = self.pos_encoder(x)  # ✅ Positional Encoding 적용
        x = self.transformer_encoder(x)
        x = self.fc_out(x)
        return x
