import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.SelfAttention_Family import AttentionLayer, FullAttention


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()
        xn = (x - mu) / sigma
        if self.affine:
            xn = xn * self.weight + self.bias
        return xn, (mu, sigma)

    def denormalize(self, y, stats):
        mu, sigma = stats
        return y * sigma + mu


class CrossPhaseRoutingLayer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_routers: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
        period_len: int = 24,
        use_pos_embed: bool = False,
        pos_dropout: float = 0.0,
    ):
        super().__init__()
        self.use_pos_embed = use_pos_embed
        self.period_len = period_len

        self.router = nn.Parameter(torch.randn(num_routers, latent_dim))
        nn.init.trunc_normal_(self.router, std=0.02)

        if self.use_pos_embed:
            self.pos_embedding = nn.Parameter(torch.zeros(period_len, latent_dim))
            nn.init.trunc_normal_(self.pos_embedding, std=0.02)
            self.pos_dropout = nn.Dropout(pos_dropout)

        self.router_sender = AttentionLayer(
            FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
            latent_dim,
            num_heads,
        )
        self.router_receiver = AttentionLayer(
            FullAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
            latent_dim,
            num_heads,
        )

        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.GELU(),
            nn.Linear(4 * latent_dim, latent_dim),
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, z):
        B, C, L, D = z.shape
        x = z.view(B * C, L, D)

        if self.use_pos_embed:
            if L == self.period_len:
                pe = self.pos_embedding.unsqueeze(0).expand(B * C, -1, -1)
            elif L < self.period_len:
                pe = self.pos_embedding[:L, :].unsqueeze(0).expand(B * C, -1, -1)
            else:
                repeat_factor = (L + self.period_len - 1) // self.period_len
                expanded_pe = self.pos_embedding.repeat(repeat_factor, 1)
                pe = expanded_pe[:L, :].unsqueeze(0).expand(B * C, -1, -1)
            x = self.pos_dropout(x + pe)

        batch_router = self.router.unsqueeze(0).expand(B * C, -1, -1)
        router_buffer, _ = self.router_sender(batch_router, x, x, attn_mask=None)
        router_receive, _ = self.router_receiver(x, router_buffer, router_buffer, attn_mask=None)

        out = self.norm1(x + self.dropout_layer(router_receive))
        out = self.norm2(out + self.dropout_layer(self.mlp(out)))
        return out.view(B, C, L, D)


class PhaseEmbedding(nn.Module):
    def __init__(self, p_in: int, latent_dim: int, hidden: int = 32, use_mlp: bool = False, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        if use_mlp:
            self.projection = nn.Sequential(
                nn.Linear(p_in, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, latent_dim),
            )
        else:
            self.projection = nn.Linear(p_in, latent_dim)

    def forward(self, phase_series):
        return self.norm(self.projection(phase_series))


class PhasePredictor(nn.Module):
    def __init__(self, p_out: int, latent_dim: int, hidden: int, use_mlp: bool = False, dropout: float = 0.0):
        super().__init__()
        self.use_mlp = use_mlp
        if use_mlp:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                nn.Linear(hidden, p_out),
            )
        else:
            self.decoder = nn.Linear(latent_dim, p_out)
            self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, z):
        if self.use_mlp:
            return self.decoder(z)
        return self.decoder(self.dropout(z))


class CrossPhaseRoutingUnit(nn.Module):
    def __init__(
        self,
        apply_in_proj: bool,
        apply_out_proj: bool,
        num_periods_input: int,
        latent_dim: int,
        phase_attn_heads: int,
        phase_attn_dropout: float,
        period_len: int,
        phase_num_routers: int = 8,
        phase_use_pos_embed: bool = False,
        phase_pos_dropout: float = 0.0,
    ):
        super().__init__()
        self.apply_in_proj = apply_in_proj

        self.in_proj = (
            nn.Sequential(nn.Linear(num_periods_input, latent_dim), nn.LayerNorm(latent_dim))
            if apply_in_proj
            else None
        )
        self.interact = CrossPhaseRoutingLayer(
            latent_dim=latent_dim,
            num_routers=phase_num_routers,
            num_heads=phase_attn_heads,
            dropout=phase_attn_dropout,
            period_len=period_len,
            use_pos_embed=phase_use_pos_embed,
            pos_dropout=phase_pos_dropout,
        )
        self.out_proj = nn.Linear(latent_dim, num_periods_input) if apply_out_proj else None

    def forward(self, phase_series, z_prev=None):
        if self.apply_in_proj:
            z_curr = self.in_proj(phase_series)
            z = z_curr if z_prev is None else z_prev + z_curr
        else:
            if z_prev is None:
                raise ValueError("z_prev must be provided when apply_in_proj is False")
            z = z_prev

        z = self.interact(z)
        y_phase_steps = self.out_proj(z) if self.out_proj is not None else None
        return z, y_phase_steps


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        self.latent_dim = configs.latent_dim
        self.phase_encoder_hidden = configs.phase_encoder_hidden
        self.predictor_hidden = configs.predictor_hidden

        self.phase_attn_heads = configs.phase_attn_heads
        self.phase_attn_dropout = configs.phase_attn_dropout
        self.phase_num_routers = configs.phase_num_routers
        self.phase_use_pos_embed = configs.phase_use_pos_embed
        self.phase_pos_dropout = configs.phase_pos_dropout

        self.num_periods_input = (self.seq_len + self.period_len - 1) // self.period_len
        self.num_periods_output = (self.pred_len + self.period_len - 1) // self.period_len
        self.total_len_in = self.num_periods_input * self.period_len
        self.pad_seq_len = self.total_len_in - self.seq_len

        self.use_revin = configs.use_revin
        if self.use_revin:
            self.revin = RevIN(
                num_features=self.enc_in,
                eps=configs.revin_eps,
                affine=configs.revin_affine,
            )

        self.phase_layers = configs.phase_layers
        self.embedding = PhaseEmbedding(
            p_in=self.num_periods_input,
            latent_dim=self.latent_dim,
            hidden=self.phase_encoder_hidden,
            use_mlp=configs.phase_encoder_use_mlp,
            dropout=configs.phase_encoder_dropout,
        )

        routing_units = []
        if self.phase_layers == 1:
            routing_units.append(
                CrossPhaseRoutingUnit(
                    apply_in_proj=False,
                    apply_out_proj=False,
                    num_periods_input=self.num_periods_input,
                    latent_dim=self.latent_dim,
                    phase_attn_heads=self.phase_attn_heads,
                    phase_attn_dropout=self.phase_attn_dropout,
                    period_len=self.period_len,
                    phase_num_routers=self.phase_num_routers,
                    phase_use_pos_embed=self.phase_use_pos_embed,
                    phase_pos_dropout=self.phase_pos_dropout,
                )
            )
        else:
            for li in range(self.phase_layers):
                is_first = li == 0
                is_last = li == self.phase_layers - 1
                routing_units.append(
                    CrossPhaseRoutingUnit(
                        apply_in_proj=not is_first,
                        apply_out_proj=not is_last,
                        num_periods_input=self.num_periods_input,
                        latent_dim=self.latent_dim,
                        phase_attn_heads=self.phase_attn_heads,
                        phase_attn_dropout=self.phase_attn_dropout,
                        period_len=self.period_len,
                        phase_num_routers=self.phase_num_routers,
                        phase_use_pos_embed=self.phase_use_pos_embed,
                        phase_pos_dropout=self.phase_pos_dropout,
                    )
                )
        self.routing_layers = nn.ModuleList(routing_units)

        self.predictor = PhasePredictor(
            p_out=self.num_periods_output,
            latent_dim=self.latent_dim,
            hidden=self.predictor_hidden,
            use_mlp=configs.predictor_use_mlp,
            dropout=configs.predictor_dropout,
        )

    @staticmethod
    def _to_phase_series(x_periods):
        return x_periods.permute(0, 1, 3, 2).contiguous()

    @staticmethod
    def _from_phase_steps_to_periods(y_phase_steps):
        return y_phase_steps.permute(0, 1, 3, 2).contiguous()

    def _forecast(self, x_enc):
        if self.use_revin:
            x_in, stats = self.revin.normalize(x_enc)
        else:
            x_in = x_enc.float()
            stats = None

        x = x_in.permute(0, 2, 1)
        B, C, _ = x.shape
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode="circular")

        x_periods = x.view(B, C, self.num_periods_input, self.period_len)
        phase_series = self._to_phase_series(x_periods)

        z = self.embedding(phase_series)
        phase_series_cur = phase_series

        for layer_index, unit in enumerate(self.routing_layers):
            z, y_phase_steps_p_in = unit(phase_series_cur, z)
            if layer_index < len(self.routing_layers) - 1:
                phase_series_cur = y_phase_steps_p_in

        y_phase_steps = self.predictor(z)
        y_periods = self._from_phase_steps_to_periods(y_phase_steps)
        y_full = y_periods.reshape(B, C, -1)[..., : self.pred_len]
        y_hat = y_full.permute(0, 2, 1)

        if self.use_revin:
            y_hat = self.revin.denormalize(y_hat, stats)
        return y_hat

    def forward(self, input, x_exo_con=None, x_exo_dis=None, y_exo_con=None, y_exo_dis=None):
        return self._forecast(input)
