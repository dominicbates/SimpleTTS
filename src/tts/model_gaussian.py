import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ── Vocabulary ────────────────────────────────────────────────────────────────
def build_vocab(extra_tokens=None, single_unknown = True):
    arpabet = [
        'AA0','AA1','AA2','AE0','AE1','AE2','AH0','AH1','AH2',
        'AO0','AO1','AO2','AW0','AW1','AW2','AY0','AY1','AY2',
        'B','CH','D','DH','EH0','EH1','EH2','ER0','ER1','ER2',
        'EY0','EY1','EY2','F','G','HH','IH0','IH1','IH2',
        'IY0','IY1','IY2','JH','K','L','M','N','NG',
        'OW0','OW1','OW2','OY0','OY1','OY2','P','R','S','SH',
        'T','TH','UH0','UH1','UH2','UW0','UW1','UW2','V',
        'W','Y','Z','ZH',
    ]
    special = ['<PAD>', '<UNK>', '|', '<EXCLAMATION>', '<QUESTION>',
               '<COMMA>', '<PERIOD>', '<COLON>', '<SEMICOLON>']
    if extra_tokens:
        special += extra_tokens

    vocab = {tok: i for i, tok in enumerate(special + arpabet)}
    offset = len(vocab)

    # Using single unknown token
    if single_unknown:
        vocab['<UNK_WORD>'] = offset
        offset += 1  # Not needed since last one

    else:
        for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            vocab[f'<UNK_{ch}>'] = offset
            offset += 1

    return vocab


# ── Positional Encoding ───────────────────────────────────────────────────────
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)


# ── Transformer Encoder ───────────────────────────────────────────────────────
class PhonemeTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, token_ids, padding_mask=None):
        x = self.embedding(token_ids)
        x = self.pos_enc(x)
        return self.transformer(x, src_key_padding_mask=padding_mask)


# ── Output Heads ──────────────────────────────────────────────────────────────
class SpectrogramHead(nn.Module):
    def __init__(self, d_model: int, n_mels: int, patch_frames: int):
        super().__init__()
        self.n_mels       = n_mels
        self.patch_frames = patch_frames
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_mels * patch_frames),
        )

    def forward(self, x):
        B, S, _ = x.shape
        return self.net(x).view(B, S, self.n_mels, self.patch_frames)


class DurationHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        return F.softplus(self.net(x)).squeeze(-1)  # (batch, seq_len), always positive


# ── Spectrogram Assembly ──────────────────────────────────────────────────────
# def assemble_spectrogram(
#     patches:       torch.Tensor,  # (S, n_mels, patch_frames)
#     positions:     torch.Tensor,  # (S,)  centre of each token in frames
#     sigma:         torch.Tensor,  # scalar — shared learned Gaussian width
#     silence:       torch.Tensor,  # scalar — shared learned silence offset
#     target_frames: int,
# ) -> torch.Tensor:
def assemble_spectrogram(
    patches:       torch.Tensor,   # (S, n_mels, patch_frames)
    positions:     torch.Tensor,   # (S,)
    sigma:         torch.Tensor,
    silence:       torch.Tensor,
    target_frames: int,
    padding_mask:  torch.Tensor = None,  # (S,) True = padding
) -> torch.Tensor:
    """
    Differentiable spectrogram assembly via Gaussian splatting.

    Each token i contributes its patch to the canvas via a Gaussian window
    centred at positions[i] with shared width sigma. Contributions are summed
    (not averaged), so silence regions naturally tend to zero. The learned
    silence scalar is then subtracted globally, so those near-zero regions
    settle at the true silence level.

    Returns: (n_mels, target_frames)
    """
    S, n_mels, P = patches.shape
    device = patches.device

    # Absolute frame centre for each patch column p of token i:
    #   positions[i] - P/2 + p + 0.5
    # shape: (S, P)
    patch_centres = positions.unsqueeze(1) + (
        torch.arange(P, device=device).float() - P / 2.0 + 0.5
    )

    # Output frame indices: (T,)
    frame_idx = torch.arange(target_frames, device=device).float()

    # Gaussian weights: (S, P, T)
    diff    = patch_centres.unsqueeze(2) - frame_idx.unsqueeze(0).unsqueeze(0)
    # weights = torch.exp(-0.5 * diff ** 2 / sigma ** 2)

    # # Weighted sum over all tokens and patch columns → (n_mels, T)
    # patches_t = patches.permute(0, 2, 1)                                         # (S, P, n_mels)
    # canvas    = (patches_t.unsqueeze(3) * weights.unsqueeze(2)).sum(dim=(0, 1))  # (n_mels, T)

    # # Subtract learned silence: silent frames (sum ≈ 0) become ≈ -silence,
    # # which the model learns to match the true silence value in the target.
    # return canvas - silence

    weights = torch.exp(-0.5 * diff ** 2 / sigma ** 2)   # (S, P, T)

    # Zero out contributions from padding tokens
    if padding_mask is not None:
        weights = weights * (~padding_mask).float().unsqueeze(1).unsqueeze(2)

    patches_t = patches.permute(0, 2, 1)                                         # (S, P, n_mels) 
    canvas    = (patches_t.unsqueeze(3) * weights.unsqueeze(2)).sum(dim=(0, 1))  # (n_mels, T)
    return canvas - silence


    

# ── Loss ─────────────────────────────────────────────────────────────────────
def spectrogram_loss(
    pred_spec:    torch.Tensor,  # (n_mels, total_frames)
    target_spec:  torch.Tensor,  # (n_mels, target_frames)
    sigma:        torch.Tensor,  # scalar
    sigma_weight: float = 0.01,
) -> torch.Tensor:
    """
    L1 reconstruction loss over the overlapping region, plus a small penalty
    on sigma to encourage sharpness once positions have converged.
    """
    T = min(pred_spec.shape[1], target_spec.shape[1])
    loss_recon = F.l1_loss(pred_spec[:, :T], target_spec[:, :T])
    loss_sigma = sigma_weight * sigma  # nudges sigma down without fighting reconstruction
    return loss_recon + loss_sigma


# ── Full Model ────────────────────────────────────────────────────────────────
class PhonemeToSpectrogram(nn.Module):
    """
    token ids → mel spectrogram

    Each phoneme token predicts:
      - a 2D patch  (n_mels × patch_frames)  — spectral content
      - a duration  (scalar, frames)          — used to derive centre position

    Two global learned scalars are shared across all tokens:
      - sigma    — Gaussian placement width (starts wide, shrinks as positions converge)
      - silence  — subtracted from the canvas so zero-coverage regions = silence
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int         = 256,
        nhead: int           = 4,
        num_layers: int      = 4,
        dim_feedforward: int = 1024,
        n_mels: int          = 100,
        patch_frames: int    = 16,
        dropout: float       = 0.1,
        sigma_init: float    = 5.0,   # starting Gaussian width in frames
        silence_init: float  = 8.0,   # starting silence offset (positive → shifts canvas down)
        sigma_weight: float  = 0.01,
    ):
        super().__init__()
        self.patch_frames = patch_frames
        self.n_mels       = n_mels
        self.sigma_weight = sigma_weight

        self.encoder   = PhonemeTransformerEncoder(
            vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout
        )
        self.spec_head = SpectrogramHead(d_model, n_mels, patch_frames)
        self.dur_head  = DurationHead(d_model)

        # Global learned scalars — stored in log space so they stay positive
        # and the optimiser moves symmetrically on a scale axis.
        self.log_sigma   = nn.Parameter(torch.tensor(math.log(sigma_init)))
        self.log_silence = nn.Parameter(torch.tensor(math.log(silence_init)))

    @property
    def sigma(self) -> torch.Tensor:
        return self.log_sigma.exp()

    @property
    def silence(self) -> torch.Tensor:
        return self.log_silence.exp()

    def forward(self, token_ids, padding_mask=None):
        emb       = self.encoder(token_ids, padding_mask)          # (B, S, D)
        patches   = self.spec_head(emb)                            # (B, S, n_mels, P)
        durations = self.dur_head(emb)                             # (B, S), always positive

        # Prevent padding tokens from contributing to positions or canvas
        if padding_mask is not None:
            durations = durations.masked_fill(padding_mask, 0.0)
            
        # Centre position of each token = start + half duration
        starts    = torch.cumsum(durations, dim=-1) - durations    # (B, S)
        positions = starts + durations / 2.0                       # (B, S)

        return patches, positions, padding_mask
        

    def compute_loss(
        self,
        token_ids:    torch.Tensor,   # (batch, seq_len)
        target_spec:  torch.Tensor,   # (batch, n_mels, target_frames)
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        patches, positions = self.forward(token_ids, padding_mask)
        target_frames = target_spec.shape[2]

        return torch.stack([
            spectrogram_loss(
                assemble_spectrogram(patches[b], positions[b], self.sigma, self.silence, target_frames),
                target_spec[b],
                self.sigma,
                self.sigma_weight,
            )
            for b in range(token_ids.shape[0])
        ]).mean()

    @torch.no_grad()
    def synthesise(
        self,
        token_ids:    torch.Tensor,   # (1, seq_len)
        max_frames:   int = 1024,#2048, #1024,
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Inference: returns assembled spectrogram (n_mels, total_frames)."""
        patches, positions = self.forward(token_ids, padding_mask)
        return assemble_spectrogram(patches[0], positions[0], self.sigma, self.silence, max_frames)
