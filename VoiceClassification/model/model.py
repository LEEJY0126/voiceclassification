import torch
import torch.nn as nn


class STFTFrontend(nn.Module):
    """
    raw waveform → magnitude STFT

    Input : [B, 16000]
    Output: [B, 1, T, F] = [B, 1, 63, 513]
    """
    def __init__(self, n_fft=1024, hop_length=256):
        super().__init__()
        self.n_fft      = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, x):                        # [B, 16000]
        specs = torch.stft(
            x.reshape(-1, x.shape[-1]),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )                                        # [B, 513, 63] complex
        mag = specs.abs()                        # [B, 513, 63]
        mag = mag.permute(0, 2, 1).unsqueeze(1) # [B, 1, 63, 513]
        return mag


class CNNTimeCompressor(nn.Module):
    """
    시간 축을 먼저 압축하면서 채널 feature 추출

    Input : [B, 1,   63, 513]
    Block1: [B, 32,  32, 257]
    Block2: [B, 64,  16, 129]
    Block3: [B, 128,  8,  65]
    Pool  : [B, 128,  1,  64]
    Output: [B, 64, 128]       ← 64개 주파수 토큰, 각 128-dim
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: [B, 1,   63, 513] → [B, 32,  32, 257]
            nn.Conv2d(1,   32,  kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Block 2: [B, 32,  32, 257] → [B, 64,  16, 129]
            nn.Conv2d(32,  64,  kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 3: [B, 64,  16, 129] → [B, 128,  8,  65]
            nn.Conv2d(64,  128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # time 8→1, freq 65→64
            nn.AdaptiveAvgPool2d((1, 64)),       # [B, 128, 1, 64]
        )

    def forward(self, x):                        # [B, 1, 63, 513]
        x = self.net(x)                          # [B, 128, 1, 64]
        x = x.squeeze(2).permute(0, 2, 1)        # [B, 64, 128]
        return x


class FreqTransformer(nn.Module):
    """
    64개 주파수 토큰 간 관계 포착

    Input : [B, 64, 128]
    Output: [B, 64, 128]  ← 다른 주파수 정보가 반영된 embedding
    """
    def __init__(self, d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):                        # [B, 64, 128]
        x = x + self.pos_emb                     # 주파수 위치 정보 추가
        x = self.transformer(x)                  # [B, 64, 128]
        return x


class VoiceClassifier(nn.Module):
    """
    End-to-End 음성 명령 분류기

    Input : raw waveform [B, 16000]
    Output: logits [B, 4]
        0 → 기타
        1 → "라붕아"
        2 → "가"
        3 → "멈춰"

    Pipeline:
        [B, 16000]
          → STFTFrontend        [B, 1, 63, 513]
          → CNNTimeCompressor   [B, 64, 128]
          → FreqTransformer     [B, 64, 128]
          → Global Avg Pool     [B, 128]
          → FC                  [B, 4]
    """
    def __init__(self):
        super().__init__()
        self.frontend    = STFTFrontend()
        self.cnn         = CNNTimeCompressor()
        self.transformer = FreqTransformer()
        self.classifier  = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 4),
        )

    def forward(self, x):                        # [B, 16000]
        x = self.frontend(x)                     # [B, 1,  63, 513]
        x = self.cnn(x)                          # [B, 64, 128]
        x = self.transformer(x)                  # [B, 64, 128]
        x = x.mean(dim=1)                        # [B, 128]
        x = self.classifier(x)                   # [B, 4]
        return x