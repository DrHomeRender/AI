import torch
import torch.nn as nn
import math

# (단일 입력 시퀀스) → Encoder → Flatten → Regression
class PositionalEncoding(nn.Module):
    """포지셔널 인코딩"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

# 가장 기본이 되는 계산단위
class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn

# 여러개의 ScaledDotProductAttention 을 병렬로 수행하고 concat+projection 하는구조 머리를 여러개 둬서 다양한 관점을 보게하는 구조
class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = self.d_v = d_model // n_head
        self.n_head = n_head

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        # (batch, head, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_head, self.d_k).transpose(1, 2)

        out, attn = self.attention(Q, K, V, mask)

        # (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(out)


class PositionwiseFeedForward(nn.Module):
    """Position-wise FFN"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Transformer Encoder Layer"""
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed Forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class TransformerModel(nn.Module):
    """전체 Transformer 시계열 예측 모델"""
    def __init__(self, input_dim, output_dim, d_model=64, n_head=4, d_ff=128, num_layers=2, dropout=0.1, seq_len=10):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_head, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(d_model * seq_len, output_dim)
        self.seq_len = seq_len

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)

        x = x.reshape(x.size(0), -1)  # Flatten
        return self.output_layer(x)


# 테스트용 샘플 입력
if __name__ == "__main__":
    batch_size = 16
    seq_len = 10
    input_dim = 1
    output_dim = 1

    sample_input = torch.randn(batch_size, seq_len, input_dim)
    model = TransformerModel(input_dim=input_dim, output_dim=output_dim, seq_len=seq_len)
    output = model(sample_input)

    print("Transformer 출력 shape:", output.shape)
