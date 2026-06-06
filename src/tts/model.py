import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Seems to be the case in the spctra
SILENCE_VALUE = -8.0


# ── Vocabulary ────────────────────────────────────────────────────────────────

def build_vocab(extra_tokens=None):
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


# class PositionHead(nn.Module):
#     """
#     Predicts absolute start position (in frames) for each token.
#     Output is unbounded (-inf to +inf); monotonicity is enforced via a
#     separate penalty loss rather than by construction.
#     """
#     def __init__(self, d_model: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(d_model, d_model // 2),
#             nn.GELU(),
#             nn.Linear(d_model // 2, 1),
#         )

#     def forward(self, x):
#         return self.net(x).squeeze(-1)   # (batch, seq_len), unbounded

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


# ── Tent Window ───────────────────────────────────────────────────────────────

def make_tent_window(patch_frames: int, device) -> torch.Tensor:
    """Triangular window: 0 → 1 → 0, for smooth patch blending."""
    t = torch.linspace(0, 1, patch_frames, device=device)
    return 1.0 - torch.abs(2.0 * t - 1.0)


# ── Spectrogram Assembly ──────────────────────────────────────────────────────

# def assemble_spectrogram(
#     patches: torch.Tensor,    # (seq_len, n_mels, patch_frames)
#     positions: torch.Tensor,  # (seq_len,) absolute start frame per token, unbounded
#     target_frames: int,
# ) -> torch.Tensor:
#     """
#     Place each token's patch at its predicted absolute position and blend
#     overlaps using a tent-windowed weighted average.

#     Patches extending before t=0 are trimmed on the left.
#     Patches extending past target_frames are included in the canvas; the loss
#     function penalises that region against silence without any hard clamping.

#     Returns: (n_mels, total_frames) where total_frames >= target_frames.
#     """
#     seq_len, n_mels, patch_frames = patches.shape
#     device = patches.device

#     last_end     = int((positions[-1] + patch_frames).item())
#     total_frames = max(target_frames, last_end)

#     window     = make_tent_window(patch_frames, device)
#     # spec_sum   = torch.zeros(n_mels, total_frames, device=device)
    
#     # Fill with -8 which is roughly silence on this scale
#     spec_sum = torch.full((n_mels, total_frames), SILENCE_VALUE, device=device)

#     weight_sum = torch.zeros(total_frames, device=device)

#     for i in range(seq_len):
#         t0_int = int(positions[i].item())

#         # Left-edge trim: skip frames that fall before t=0
#         patch_offset = max(0, -t0_int)
#         canvas_start = max(0, t0_int)
#         canvas_end   = min(t0_int + patch_frames, total_frames)
#         w_len        = canvas_end - canvas_start
#         if w_len <= 0:
#             continue

#         patch_slice  = patches[i, :, patch_offset : patch_offset + w_len]
#         window_slice = window[patch_offset : patch_offset + w_len]

#         spec_sum  [:, canvas_start:canvas_end] += patch_slice * window_slice.unsqueeze(0)
#         weight_sum[canvas_start:canvas_end]    += window_slice

#     # Weighted average: normalise by accumulated window weights
#     # spec = spec_sum / weight_sum.clamp(min=1e-8).unsqueeze(0)
    
#     # Replace the final normalisation line in assemble_spectrogram
#     mask = weight_sum > 0
#     spec = spec_sum.clone()
#     spec[:, mask] = spec_sum[:, mask] / weight_sum[mask].unsqueeze(0)
#     # frames with no patches stay at -8.0 (silence)
    
#     return spec  # (n_mels, total_frames)



def assemble_spectrogram(
    patches: torch.Tensor,     # (S, n_mels, P)
    positions: torch.Tensor,   # (S,)
    target_frames: int,
):
    """
    Fully differentiable spectrogram assembly.

    Replaces discrete splatting with continuous alignment.
    """

    S, n_mels, P = patches.shape
    device = patches.device

    # --- output canvas ---
    t = torch.arange(target_frames, device=device).float()  # (T,)

    # We'll accumulate contributions here
    spec = torch.zeros(n_mels, target_frames, device=device)
    weight = torch.zeros(target_frames, device=device)

    # --- temperature controls sharpness (can anneal) ---
    sigma = 5.0 # 2.0  # try 5.0 early training, then anneal down

    for i in range(S):

        # (T,) continuous center for this token patch start
        center = positions[i]  # scalar float tensor

        # local offsets inside patch
        p = torch.arange(P, device=device).float()  # (P,)

        # broadcast to (P, T)
        # each patch frame has its own center in time
        centers = center + p[:, None]  # (P, 1)

        # compute soft assignment to all output frames
        # (P, T)
        dist2 = (t[None, :] - centers) ** 2
        w = torch.exp(-dist2 / (2 * sigma * sigma))

        # optional: normalize per patch frame (prevents energy explosion)
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)

        # accumulate
        # patches[i]: (n_mels, P)
        # w: (P, T)
        spec += torch.einsum("mp,pt->mt", patches[i], w)

        weight += w.sum(dim=0)

    # normalize overlapping contributions
    spec = spec / (weight.unsqueeze(0) + 1e-8)

    spec -= 8 # Baseline to -8 
    return spec

    
# ── Loss ──────────────────────────────────────────────────────────────────────

# def spectrogram_loss(
#     pred_spec: torch.Tensor,    # (n_mels, total_frames)
#     target_spec: torch.Tensor,  # (n_mels, target_frames)
# ) -> torch.Tensor:
#     """
#     L1Loss loss (MAE) between predicted and target spectrogram.

#     Frames within target_frames: compared directly.
#     Frames beyond target_frames: penalised against silence (0.0).
#     """
#     n_mels, total_frames = pred_spec.shape
#     target_frames        = target_spec.shape[1]

#     if total_frames <= target_frames:
#         # return F.mse_loss(pred_spec, target_spec[:, :total_frames])
#         return torch.nn.L1Loss(pred_spec, target_spec[:, :total_frames])

#     # loss_in  = F.mse_loss(pred_spec[:, :target_frames], target_spec)
#     # loss_out = F.mse_loss(
#     loss_in  = torch.nn.L1Loss(pred_spec[:, :target_frames], target_spec)
#     loss_out = torch.nn.L1Loss(
#         pred_spec[:, target_frames:],
#         torch.zeros(n_mels, total_frames - target_frames, device=pred_spec.device),
#     )
#     return loss_in + loss_out

def spectrogram_loss(
    pred_spec: torch.Tensor,    # (n_mels, total_frames)
    target_spec: torch.Tensor,  # (n_mels, target_frames)
) -> torch.Tensor:
    n_mels, total_frames = pred_spec.shape
    target_frames        = target_spec.shape[1]

    if total_frames <= target_frames:
        return F.l1_loss(pred_spec, target_spec[:, :total_frames])

    loss_in  = F.l1_loss(pred_spec[:, :target_frames], target_spec)
    loss_out = F.l1_loss(
        pred_spec[:, target_frames:],
        torch.zeros(n_mels, total_frames - target_frames, device=pred_spec.device),
    )
    # Compare to silence
    loss_out = F.l1_loss(
        pred_spec[:, target_frames:],
        torch.full((n_mels, total_frames - target_frames), SILENCE_VALUE, device=pred_spec.device),
    )
    return loss_in + loss_out


def monotonicity_loss(positions: torch.Tensor) -> torch.Tensor:
    """
    Penalises any token whose predicted position is not strictly greater than
    the previous token's position. Zero when fully ordered, positive otherwise.

    positions: (batch, seq_len)
    """
    diffs = positions[:, 1:] - positions[:, :-1]   # (batch, seq_len-1)
    return F.relu(-diffs).mean()


# ── Full Model ────────────────────────────────────────────────────────────────

class PhonemeToSpectrogram(nn.Module):
    """
    token ids → (patches, positions)

    patches:   (batch, seq_len, n_mels, patch_frames)
    positions: (batch, seq_len)  absolute start frame per token, unbounded
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
        # monotonicity_weight: float = 1.0,
    ):
        super().__init__()
        self.patch_frames         = patch_frames
        self.n_mels               = n_mels
        # self.monotonicity_weight  = monotonicity_weight

        self.encoder   = PhonemeTransformerEncoder(
            vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout
        )
        self.spec_head = SpectrogramHead(d_model, n_mels, patch_frames)
        # self.pos_head  = PositionHead(d_model)
        self.dur_head = DurationHead(d_model)

    # def forward(self, token_ids, padding_mask=None):
    #     emb       = self.encoder(token_ids, padding_mask)
    #     patches   = self.spec_head(emb)
    #     positions = self.pos_head(emb)
    #     return patches, positions

    def forward(self, token_ids, padding_mask=None):
        emb       = self.encoder(token_ids, padding_mask)
        patches   = self.spec_head(emb)
        durations = self.dur_head(emb)                          # (B, S) positive
        positions = torch.cumsum(durations, dim=-1) - durations # (B, S) start positions
        return patches, positions
        
    # def compute_loss(
    #     self,
    #     token_ids: torch.Tensor,      # (batch, seq_len)
    #     target_spec: torch.Tensor,    # (batch, n_mels, target_frames)
    #     padding_mask: torch.Tensor = None,
    # ) -> torch.Tensor:
    #     patches, positions = self.forward(token_ids, padding_mask)
    #     target_frames = target_spec.shape[2]

    #     loss_spec = torch.stack([
    #         spectrogram_loss(
    #             assemble_spectrogram(patches[b], positions[b], target_frames),
    #             target_spec[b],
    #         )
    #         for b in range(token_ids.shape[0])
    #     ]).mean()

    #     loss_mono = monotonicity_loss(positions)

    #     return loss_spec + self.monotonicity_weight * loss_mono

    def compute_loss(
        self,
        token_ids: torch.Tensor,      # (batch, seq_len)
        target_spec: torch.Tensor,    # (batch, n_mels, target_frames)
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        patches, positions = self.forward(token_ids, padding_mask)
        target_frames = target_spec.shape[2]
    
        return torch.stack([
            spectrogram_loss(
                assemble_spectrogram(patches[b], positions[b], target_frames),
                target_spec[b],
            )
            for b in range(token_ids.shape[0])
        ]).mean()

    
    @torch.no_grad()
    def synthesise(
        self,
        token_ids: torch.Tensor,      # (1, seq_len)
        max_frames: int = 1024,#2048#, roughly 10 seconds
        padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Inference: returns assembled spectrogram (n_mels, total_frames)."""
        patches, positions = self.forward(token_ids, padding_mask)
        return assemble_spectrogram(patches[0], positions[0], max_frames)


# ── Sanity Check ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    vocab = build_vocab()
    model = PhonemeToSpectrogram(vocab_size=len(vocab))

    tokens = ['HH', 'AY1', '|', '<EXCLAMATION>']
    ids    = torch.tensor([[vocab.get(t, vocab['<UNK>']) for t in tokens]])

    patches, positions = model(ids)
    print(f'patches:   {patches.shape}')    # (1, 4, 80, 16)
    print(f'positions: {positions}')        # unbounded, ideally increasing

    target = torch.randn(1, 80, 100)
    loss   = model.compute_loss(ids, target)
    print(f'loss: {loss.item():.4f}')

    spec = model.synthesise(ids, max_frames=200)
    print(f'spec: {spec.shape}')            # (80, >=200)