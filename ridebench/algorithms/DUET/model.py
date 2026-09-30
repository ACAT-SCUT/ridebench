import torch.nn as nn
from einops import rearrange

from .layers.linear_extractor_cluster import LinearExtractorCluster
from .layers.masked_attention import (
    AttentionLayer,
    Encoder,
    EncoderLayer,
    FullAttention,
    MahalanobisMask,
)


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.cluster = LinearExtractorCluster(configs)
        self.CI = configs.CI
        self.n_vars = configs.enc_in
        self.mask_generator = MahalanobisMask(configs.seq_len)
        self.channel_transformer = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            True,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.linear_head = nn.Sequential(
            nn.Linear(configs.d_model, configs.pred_len),
            nn.Dropout(configs.fc_dropout),
        )

    def forward(self, x_endo, *args, **kwargs):
        if self.CI:
            channel_independent_input = rearrange(x_endo, "b l n -> (b n) l 1")
            reshaped_output, importance_loss = self.cluster(channel_independent_input)
            temporal_feature = rearrange(
                reshaped_output,
                "(b n) l 1 -> b l n",
                b=x_endo.shape[0],
                n=x_endo.shape[2],
            )
        else:
            temporal_feature, importance_loss = self.cluster(x_endo)

        temporal_feature = rearrange(temporal_feature, "b d n -> b n d")
        if self.n_vars > 1:
            changed_input = rearrange(x_endo, "b l n -> b n l")
            channel_mask = self.mask_generator(changed_input)
            channel_group_feature, _ = self.channel_transformer(
                x=temporal_feature,
                attn_mask=channel_mask,
            )
            output = self.linear_head(channel_group_feature)
        else:
            output = self.linear_head(temporal_feature)

        output = rearrange(output, "b n d -> b d n")
        output = self.cluster.revin(output, "denorm")
        return output, importance_loss
