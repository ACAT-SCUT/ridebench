import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..covariate_adapter import build_covariate_adapter


class Conv1dNonOverlapping(nn.Module):
    def __init__(self, in_channels, out_channels, input_length, kernel_size, groups=1):
        super().__init__()
        extra_length = (kernel_size - (input_length % kernel_size)) % kernel_size
        padding = extra_length // 2
        self.input_length = input_length
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels * kernel_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            padding=padding,
            groups=groups,
        )

    def forward(self, x):
        x = self.conv1d(x)
        conv_len = x.shape[-1]
        x = x.reshape(x.shape[0], self.out_channels, self.kernel_size, conv_len)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(x.shape[0], self.out_channels, conv_len * self.kernel_size)
        padding = self.input_length - x.shape[-1]
        return F.pad(x, (padding, 0))


class GroupedLinears(nn.Module):
    def __init__(self, in_d, out_d, n=16, zero_init=True, init=0.0, split_ratio=1):
        super().__init__()
        self.split_ratio = min(split_ratio, n)
        base_segment_size = n // self.split_ratio
        remainder = n % self.split_ratio
        self.segment_sizes = [
            base_segment_size + (1 if i < remainder else 0) for i in range(self.split_ratio)
        ]
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for segment_size in self.segment_sizes:
            if zero_init:
                weight = torch.zeros(segment_size, in_d, out_d)
            else:
                weight = nn.init.trunc_normal_(
                    torch.zeros(segment_size, in_d, out_d),
                    mean=0.0,
                    std=0.01,
                    a=-0.02,
                    b=0.02,
                )
            self.weights.append(nn.Parameter(weight))
            self.biases.append(nn.Parameter(torch.zeros(segment_size, 1, out_d)))
        self.init = init

    def forward(self, x):
        outputs = []
        start = 0
        for i in range(self.split_ratio):
            end = start + self.segment_sizes[i]
            x_segment = x[:, :, start:end]
            x_segment = x_segment.permute(0, 2, 1).unsqueeze(2)
            w = self.weights[i].unsqueeze(0)
            b = self.biases[i].unsqueeze(0)
            x_segment = torch.matmul(x_segment, w) + b
            x_segment = x_segment[:, :, 0, :].permute(0, 2, 1)
            outputs.append(x_segment)
            start = end
        return torch.cat(outputs, dim=2) + self.init


class GroupedLinearsAdvanced(nn.Module):
    def __init__(self, in_d, out_d, grouping, zero_init=True, init=0.0):
        super().__init__()
        self.grouping = grouping
        self.group_weights = nn.ParameterList()
        self.group_biases = nn.ParameterList()
        for group_size in grouping:
            weight = nn.Parameter(0.01 * torch.randn(group_size, in_d, out_d))
            bias = nn.Parameter(torch.zeros(group_size, 1, out_d))
            if zero_init:
                torch.nn.init.zeros_(weight)
            self.group_weights.append(weight)
            self.group_biases.append(bias)
        self.init = init

    def forward(self, x):
        x = x.permute(0, 2, 1)
        outputs = []
        start = 0
        for i, group_size in enumerate(self.grouping):
            end = start + group_size
            group_x = x[:, start:end].unsqueeze(2)
            w = self.group_weights[i].unsqueeze(0)
            b = self.group_biases[i].unsqueeze(0)
            group_x = torch.matmul(group_x, w) + b
            outputs.append(group_x.squeeze(2))
            start = end
        x = torch.cat(outputs, dim=1).permute(0, 2, 1)
        return x + self.init


class SelectiveChannelModule(nn.Module):
    def __init__(self, input_length, input_channels, expanded_channels, drop_channels=1, mlp_ratio=2):
        super().__init__()
        self.drop_channels = drop_channels
        self.attention = nn.Sequential(
            nn.Linear(input_channels, int(mlp_ratio * expanded_channels)),
            nn.GELU(),
            nn.Linear(int(mlp_ratio * expanded_channels), expanded_channels),
            nn.Sigmoid(),
        )
        self.aggregation = nn.Sequential(nn.Linear(input_length, 1), nn.ReLU())
        self.padding = nn.Parameter(torch.ones(1, 1, drop_channels), requires_grad=False)

    def forward(self, x):
        attention_scores = self.aggregation(x.permute(0, 2, 1)).permute(0, 2, 1)
        attention_scores = self.attention(attention_scores)
        if self.drop_channels > 0:
            attention_scores = torch.cat(
                [
                    attention_scores[:, :, 0 : -self.drop_channels],
                    self.padding.repeat(x.shape[0], 1, 1),
                ],
                dim=-1,
            )
        return attention_scores


def continuity_loss(tensor):
    mean = torch.mean(tensor, dim=1, keepdim=True)
    std = torch.std(tensor, dim=1, keepdim=True)
    normalized_tensor = (tensor - mean) / (std + 1e-6)
    diffs = normalized_tensor[:, 1:, :] - normalized_tensor[:, :-1, :]
    return torch.mean(diffs.square())


class DefaultPredictor(nn.Module):
    def __init__(self, d_in, d_ff, d_out, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_in, d_ff, bias=True),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_ff, d_out, bias=True),
        )

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.model(x.permute(0, 2, 1)).permute(0, 2, 1)


class SimplePredictor(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.model = nn.Linear(d_in, d_out)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.model(x.permute(0, 2, 1)).permute(0, 2, 1)


class MLPPredictor(nn.Module):
    def __init__(self, d_in, d_ff, d_out, dropout=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(d_in, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_out),
        )

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.model(x.permute(0, 2, 1)).permute(0, 2, 1)


class MeanAggregationPredictor(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return x.mean(dim=1, keepdim=True).repeat(1, self.pred_len, 1)


class IndependentLinearPredictor(nn.Module):
    def __init__(self, d_in, d_out, n_channels, split_ratio=1):
        super().__init__()
        self.model = GroupedLinears(
            d_in,
            d_out,
            n=n_channels,
            zero_init=False,
            init=0.0,
            split_ratio=split_ratio,
        )

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.model(x)


class GroupedLinearPredictor(nn.Module):
    def __init__(self, d_in, d_out, grouping):
        super().__init__()
        self.model = GroupedLinearsAdvanced(d_in, d_out, grouping=grouping, zero_init=False, init=0.0)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        return self.model(x)


class AggregatingPredictor(nn.Module):
    def __init__(self, d_in, d_out, n_channels, kernel_size=16, token_size=8, dropout=0.5, independent=True):
        super().__init__()
        groups = n_channels if independent else 1
        self.agg = nn.Conv1d(
            n_channels,
            token_size * n_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            groups=groups,
        )
        self.token_dropout = nn.Dropout1d(p=dropout, inplace=False)
        self.predictor = nn.Conv1d(d_in // kernel_size, d_out, kernel_size=token_size, stride=token_size)

    def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = x.permute(0, 2, 1)
        x = self.agg(x)
        x = self.token_dropout(x)
        x = x.permute(0, 2, 1)
        return self.predictor(x)


class CATSBackbone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.eps = 1e-7
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_channels = configs.enc_in
        self.target_dim = configs.target_dim
        self.time_emb_dim = configs.time_emb_dim
        self.in_channels_full = self.in_channels + self.time_emb_dim
        self.preserving_channels = configs.number_of_targets if configs.number_of_targets > 0 else self.in_channels

        self.preprocessing_method = "lastvalue"
        self.preprocessing_detach = True
        self.preprocessing_affine = False
        self.channel_w = nn.Parameter(torch.ones(self.preserving_channels), requires_grad=True)
        self.channel_b = nn.Parameter(torch.zeros(self.preserving_channels), requires_grad=True)

        self.F_conv_output = configs.F_conv_output
        self.F_noconv_output = configs.F_noconv_output
        self.F_gconv_output_rate = configs.F_gconv_output_rate
        self.F_lin_output = configs.F_lin_output
        self.F_id_output_rate = configs.F_id_output_rate
        self.F_emb_output = configs.F_emb_output
        self.F_gconv_output = int(self.F_gconv_output_rate * self.preserving_channels)
        self.F_id_output = int(self.preserving_channels * self.F_id_output_rate)

        self.feature_pool_desc = [
            self.F_conv_output,
            self.F_conv_output,
            self.F_noconv_output,
            self.F_noconv_output,
            self.F_gconv_output,
            self.F_lin_output,
            self.F_id_output,
            self.F_emb_output,
            self.time_emb_dim,
            self.in_channels,
        ]
        self.mlp_ratio = configs.mlp_ratio

        self.input_conv2 = nn.Conv1d(self.in_channels_full, self.F_conv_output, kernel_size=49, padding=24)
        self.input_conv2_alter = nn.Conv1d(
            self.in_channels_full,
            self.F_conv_output,
            kernel_size=193,
            padding=96,
        )
        self.input_conv = Conv1dNonOverlapping(
            self.in_channels_full,
            self.F_noconv_output,
            input_length=self.seq_len,
            kernel_size=12,
            groups=1,
        )
        self.input_conv_alter = Conv1dNonOverlapping(
            self.in_channels_full,
            self.F_noconv_output,
            input_length=self.seq_len,
            kernel_size=24,
            groups=1,
        )
        self.input_conv_grouped2 = nn.Conv1d(
            self.preserving_channels,
            self.F_gconv_output,
            kernel_size=49,
            padding=24,
            groups=self.preserving_channels,
        )
        self.input_conv_1x1 = nn.Conv1d(self.in_channels_full, self.F_lin_output, kernel_size=1, padding=0)
        self.pure_emb = nn.Parameter(torch.randn(1, self.seq_len, self.F_emb_output))

        predictor_configs = copy.deepcopy(configs)
        predictor_configs.enc_in = sum(self.feature_pool_desc)
        predictor_configs.dec_in = sum(self.feature_pool_desc)
        predictor_configs.c_out = sum(self.feature_pool_desc)
        self.predictor_name = configs.predictor
        self.Predictor = self._build_predictor(predictor_configs, configs)
        self.predictor_configs = predictor_configs

        self.continuity_beta = configs.continuity_beta
        self.gated = configs.temporal_gate
        self.generate_self_mask = GroupedLinears(
            self.seq_len,
            1,
            n=predictor_configs.enc_in,
            zero_init=True,
            init=0.0,
        )
        self.mask_generate_linspace = nn.Parameter(
            torch.linspace(1, 0, steps=self.seq_len),
            requires_grad=False,
        )

        self.channel_sparsity = configs.channel_sparsity
        self.channel_selection = SelectiveChannelModule(
            input_length=self.seq_len,
            input_channels=self.in_channels_full,
            expanded_channels=sum(self.feature_pool_desc) + self.time_emb_dim,
            drop_channels=0,
            mlp_ratio=self.mlp_ratio,
        )

        self.output_conv_1x1 = nn.Conv1d(
            predictor_configs.dec_in + self.time_emb_dim,
            self.preserving_channels,
            kernel_size=1,
            padding=0,
        )
        torch.nn.init.zeros_(self.output_conv_1x1.weight)

    def _build_predictor(self, predictor_configs, configs):
        if self.predictor_name == "Default":
            return DefaultPredictor(
                d_in=self.seq_len,
                d_ff=self.mlp_ratio * self.seq_len,
                d_out=self.pred_len,
                dropout=configs.predictor_dropout,
            )
        if self.predictor_name == "Simple":
            return SimplePredictor(d_in=self.seq_len, d_out=self.pred_len)
        if self.predictor_name == "MLP":
            return MLPPredictor(
                d_in=self.seq_len,
                d_ff=self.mlp_ratio * (self.seq_len + self.pred_len),
                d_out=self.pred_len,
                dropout=configs.predictor_dropout,
            )
        if self.predictor_name == "Mean":
            return MeanAggregationPredictor(pred_len=self.pred_len)
        if self.predictor_name == "IndependentLinear":
            return IndependentLinearPredictor(
                d_in=self.seq_len,
                d_out=self.pred_len,
                n_channels=predictor_configs.enc_in,
                split_ratio=1,
            )
        if self.predictor_name == "GroupedIndependent":
            grouping = self.feature_pool_desc[0:-1] + [1 for _ in range(self.in_channels)]
            return GroupedLinearPredictor(d_in=self.seq_len, d_out=self.pred_len, grouping=grouping)
        if self.predictor_name == "Agg":
            return AggregatingPredictor(
                d_in=self.seq_len,
                d_out=self.pred_len,
                n_channels=predictor_configs.enc_in,
                kernel_size=32,
                token_size=64,
                dropout=configs.predictor_dropout,
                independent=True,
            )
        raise ValueError(f"Unsupported CATS predictor: {self.predictor_name}")

    def gate(self, x, generated_mask, linspace, return_mask=False):
        mask_infer = -generated_mask(x)
        gate_len = x.shape[1]
        mask_infer = (
            linspace.reshape(1, gate_len, 1).repeat(mask_infer.shape[0], 1, mask_infer.shape[-1])
            * mask_infer
            + 1
        )
        mask_stop = mask_infer - mask_infer.detach() + (mask_infer > 0).int().detach()
        x = x * mask_stop
        if return_mask:
            return x, mask_stop
        return x

    def encoder(self, x):
        empty = lambda: torch.empty(  # noqa: E731
            x.shape[0],
            self.seq_len,
            0,
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat(
            [
                self.input_conv2(x.permute(0, 2, 1)).permute(0, 2, 1) if self.F_conv_output > 0 else empty(),
                self.input_conv2_alter(x.permute(0, 2, 1)).permute(0, 2, 1) if self.F_conv_output > 0 else empty(),
                self.input_conv(x.permute(0, 2, 1)).permute(0, 2, 1) if self.F_noconv_output > 0 else empty(),
                self.input_conv_alter(x.permute(0, 2, 1)).permute(0, 2, 1) if self.F_noconv_output > 0 else empty(),
                self.input_conv_grouped2(x[:, :, -self.preserving_channels :].permute(0, 2, 1)).permute(0, 2, 1)
                if self.F_gconv_output_rate > 0
                else empty(),
                self.input_conv_1x1(x.permute(0, 2, 1)).permute(0, 2, 1) if self.F_lin_output > 0 else empty(),
                *[x[:, :, -self.preserving_channels :] for _ in range(int(self.F_id_output_rate))],
                self.pure_emb.repeat(x.shape[0], 1, 1),
                x,
            ],
            dim=-1,
        )

    def predictor_(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x_pred = self.Predictor(x.clone(), x_mark_enc, x_dec, x_mark_dec)
        return torch.cat([x.clone(), x_pred], dim=1)

    def decoder(self, x, channel_attn_score_decode):
        x_resid = self.output_conv_1x1((x * channel_attn_score_decode).permute(0, 2, 1)).permute(0, 2, 1)
        output = x.clone()
        output[:, -self.pred_len :, -self.preserving_channels :] = (
            x[:, -self.pred_len :, -self.preserving_channels :] + x_resid[:, -self.pred_len :, :]
        )
        return output

    def preprocessing(self, x, x_output=None):
        x_orig = x.clone()
        if self.preprocessing_affine:
            x = x * self.channel_w + self.channel_b
        mean = x[:, -1:, :]
        std = torch.ones(1, 1, x.shape[-1], device=x.device, dtype=x.dtype)
        if x_output is None:
            x_output = x_orig.clone()
        if self.preprocessing_detach:
            x_output = (x_output - mean.detach()) / std.detach()
        else:
            x_output = (x_output - mean) / std
        if self.preprocessing_affine:
            x_output = (x_output - self.channel_b) / self.channel_w
        return x_output, mean, std

    def inverse_processing(self, x, mean, std):
        x = x.clone()
        if self.preprocessing_affine:
            x = (x - self.channel_b) / self.channel_w
        return x * std + mean

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, return_mask=False, train=False):
        x_preprocessed, mean, std = self.preprocessing(
            x_enc[:, :, -self.preserving_channels :],
            x_output=x_enc[:, :, -self.preserving_channels :],
        )
        x = torch.cat([x_enc[:, :, 0 : -self.preserving_channels], x_preprocessed], dim=-1)

        if self.time_emb_dim:
            x_enc = torch.cat([x_mark_enc, x], dim=-1)
        else:
            x_enc = x

        channel_attn_score = (
            self.channel_selection(x_enc)
            if self.channel_sparsity
            else torch.ones(
                x_enc.shape[0],
                1,
                sum(self.feature_pool_desc) + self.time_emb_dim,
                device=x_enc.device,
                dtype=x_enc.dtype,
            )
        )
        channel_attn_score_decode = torch.cat(
            [
                torch.ones_like(
                    channel_attn_score[:, :, 0 : -(self.preserving_channels + self.time_emb_dim)]
                ),
                channel_attn_score[:, :, -(self.preserving_channels + self.time_emb_dim) :],
            ],
            dim=-1,
        ).clone()
        channel_attn_score = torch.cat(
            [
                channel_attn_score[:, :, 0 : -(self.preserving_channels + self.time_emb_dim)],
                torch.ones(
                    channel_attn_score.shape[0],
                    1,
                    self.preserving_channels,
                    device=x_enc.device,
                    dtype=x_enc.dtype,
                ),
            ],
            dim=-1,
        )

        x = self.encoder(x_enc)
        x = channel_attn_score * x
        cnt_loss = continuity_loss(x) * self.continuity_beta

        if self.gated:
            if return_mask:
                x, mid_mask = self.gate(x, self.generate_self_mask, self.mask_generate_linspace, return_mask=True)
            else:
                x = self.gate(x, self.generate_self_mask, self.mask_generate_linspace)
                mid_mask = None
        else:
            mid_mask = torch.ones_like(x[:, :, :1]) if return_mask else None

        x_dec = torch.zeros(x.shape[0], self.pred_len, x.shape[-1], device=x.device, dtype=x.dtype)
        if getattr(self.predictor_configs, "label_len", 0):
            x_dec = torch.cat([x[:, -self.predictor_configs.label_len :, :], x_dec], dim=1)

        x = self.predictor_(x, x_mark_enc, x_dec, x_mark_dec)
        x_predictor_snapshot = x.detach().clone()
        x_predictor_snapshot[:, :, -self.preserving_channels :] = self.inverse_processing(
            x_predictor_snapshot[:, :, -self.preserving_channels :],
            mean,
            std,
        ).detach()

        if self.time_emb_dim:
            timestamp_embedding = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len :, :]], dim=1)
            x = torch.cat([timestamp_embedding, x], dim=-1)

        x_dec = self.decoder(x, channel_attn_score_decode)
        x = x_dec[:, -self.pred_len :, -self.preserving_channels :]
        x = self.inverse_processing(x, mean, std)

        if train:
            return x, cnt_loss
        if return_mask:
            return x, mid_mask, x_predictor_snapshot
        return x


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.core = CATSBackbone(config)
        self.pred_len = config.pred_len
        self.target_dim = config.target_dim
        self.covariate_spec = config.covariate_spec
        self.covariate_adapter = build_covariate_adapter(self.covariate_spec)
        self.use_future_exog = bool(
            self.covariate_spec.use_future_exog and self.covariate_spec.model_exog_dim > 0
        )

    def forward(
        self,
        input,
        x_exo_con=None,
        x_exo_dis=None,
        y_exo_con=None,
        y_exo_dis=None,
        return_mask=False,
        return_aux_loss=False,
    ):
        # Benchmark adaptation: notebook CATS only consumes one dense series
        # tensor. We therefore follow the YCrossLinear-style integration path:
        # append history exogenous ATS channels, and when enabled, extend the
        # time axis with zero future endogenous placeholders plus known future
        # exogenous channels.
        exog_history, exog_future = self.covariate_adapter(
            x_exo_con,
            x_exo_dis,
            y_exo_con,
            y_exo_dis,
            dtype=input.dtype,
        )
        if self.use_future_exog:
            future_endo_padding = input.new_zeros((input.shape[0], self.pred_len, input.shape[-1]))
            input = torch.cat([input, future_endo_padding], dim=1)
            if exog_history is not None:
                exog_history = torch.cat([exog_history, exog_future], dim=1)

        if exog_history is not None:
            x_enc = torch.cat([exog_history, input], dim=-1)
        else:
            x_enc = input

        if return_mask:
            return self.core(x_enc, return_mask=True)

        if return_aux_loss:
            return self.core(x_enc, train=True)

        return self.core(x_enc)
