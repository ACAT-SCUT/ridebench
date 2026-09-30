from torch import nn
import torch
from ..covariate_adapter import build_covariate_adapter
from .layers.C_EncDec import CovCausalityEncoder
from .layers.TC_EncDec import TemporalCausalityEncoder


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.stride = config.stride
        self.use_c = config.use_c
        self.use_t = config.use_t
        self.use_c_exog = config.use_c_exog
        self.use_t_exog = config.use_t_exog
        self.alpha = config.alpha
        self.beta = config.beta
        self.series_dim = config.series_dim
        self.use_future_exog = config.use_future_exog
        self.covariate_spec = config.covariate_spec
        self.covariate_adapter = build_covariate_adapter(self.covariate_spec)
        assert self.use_c or self.use_t, "At least one of use_c or use_t must be True"

        self.temporal_encoder = TemporalCausalityEncoder(
            enc_in=config.enc_in,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            series_dim=config.series_dim,
            patch_len=self.patch_len,
            stride=self.stride,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            dropout=config.dropout,
            factor=config.factor,
            activation=config.activation,
            criterion=config.criterion
        )

        self.cov_encoder = CovCausalityEncoder(
            enc_in=config.enc_in,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            series_dim=config.series_dim,
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            dropout=config.dropout,
            factor=config.factor,
            activation=config.activation,
            criterion=config.criterion

        )

    def forward(self, input, x_exo_con=None, x_exo_dis=None,
                y_exo_con=None, y_exo_dis=None):
        # Benchmark adaptation: the upstream DAG code consumes one dense float
        # exogenous block. We convert benchmark discrete IDs into dense float
        # channels here instead of feeding raw integer IDs into the encoder.
        exog_history, exog_future = self.covariate_adapter(
            x_exo_con,
            x_exo_dis,
            y_exo_con,
            y_exo_dis,
            dtype=input.dtype,
        )
        input = torch.cat([input, exog_history], dim=-1)

        temporal_causality_loss = 0
        cov_causality_loss = 0
        future_for_temporal = exog_future if self.use_future_exog else None
        future_for_cov = exog_future

        outputs = []
        weights = []
        if self.use_t:
            t_output, t_exog_output, temporal_causality_loss = self.temporal_encoder(
                input,
                future_for_temporal,
                self.use_t_exog,
            )
            outputs.append(t_output)
            weights.append(self.alpha if self.use_c else 1.0)
            if not self.use_future_exog:
                future_for_cov = t_exog_output
        if self.use_c:
            c_output, cov_causality_loss = self.cov_encoder(
                input,
                future_for_cov,
                self.use_c_exog,
            )
            outputs.append(c_output)
            weights.append(1 - self.alpha if self.use_t else 1.0)

        output = sum(weight * out for weight, out in zip(weights, outputs))
        causality_loss = self.beta * (temporal_causality_loss + cov_causality_loss)
        return output, causality_loss.reshape(1)
