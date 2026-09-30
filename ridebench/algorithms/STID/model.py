import torch
from torch import nn

from .layers.mlp import MultiLayerPerceptron


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.num_nodes = configs.num_nodes
        self.input_len = configs.input_len
        self.input_dim = configs.input_dim
        self.embed_dim = configs.embed_dim
        self.output_len = configs.output_len
        self.num_layer = configs.num_layer
        self.node_dim = configs.node_dim
        self.area_dim = configs.area_dim
        self.temp_dim_tid = configs.temp_dim_tid
        self.temp_dim_diw = configs.temp_dim_diw

        self.if_node = configs.if_node
        self.if_area = configs.if_area
        self.if_time_in_day = configs.if_T_i_D
        self.if_day_in_week = configs.if_D_i_W

        self.area_idx = configs.area_idx
        self.tid_idx = configs.tid_idx
        self.diw_idx = configs.diw_idx
        self.time_of_day_size = configs.time_of_day_size
        self.day_of_week_size = configs.day_of_week_size

        if self.if_node:
            self.node_emb = nn.Parameter(torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_area:
            self.area_emb = nn.Embedding(configs.area_size, self.area_dim)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Embedding(self.time_of_day_size, self.temp_dim_tid)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Embedding(self.day_of_week_size, self.temp_dim_diw)

        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1),
            bias=True,
        )

        self.hidden_dim = (
            self.embed_dim
            + self.node_dim * int(self.if_node)
            + self.area_dim * int(self.if_area)
            + self.temp_dim_tid * int(self.if_time_in_day)
            + self.temp_dim_diw * int(self.if_day_in_week)
        )
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1),
            bias=True,
        )

    def _expand_sample_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding.unsqueeze(-1).expand(-1, -1, self.num_nodes).unsqueeze(-1)

    def _take_last_index(self, x_exo_dis: torch.Tensor, feature_idx: int | None, modulo: int) -> torch.Tensor | None:
        if feature_idx is None or modulo <= 0:
            return None
        indices = x_exo_dis[:, -1, feature_idx].long()
        if indices.numel() > 0 and indices.min() >= 1 and indices.max() >= modulo:
            indices = indices - 1
        return indices.clamp(0, modulo - 1)

    def forward(self, x_endo, x_exo_con=None, x_exo_dis=None, y_exo_con=None, y_exo_dis=None):
        del x_exo_con, y_exo_con, y_exo_dis
        history_data = x_endo.unsqueeze(-1)
        batch_size, _, num_nodes, _ = history_data.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"STID expected {self.num_nodes} endogenous nodes, got {num_nodes}")

        input_data = history_data[..., : self.input_dim]
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        hidden_parts = [self.time_series_emb_layer(input_data)]

        if self.if_node:
            node_emb = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1)
            hidden_parts.append(node_emb.transpose(1, 2).unsqueeze(-1))

        if x_exo_dis is not None:
            if self.if_area:
                area_ids = self._take_last_index(x_exo_dis, self.area_idx, self.area_emb.num_embeddings)
                if area_ids is not None:
                    # Benchmark adaptation: STID's upstream node identity corresponds to
                    # per-sensor IDs, while this benchmark batches one area per sample.
                    # Broadcast the area_id embedding across all endogenous channels.
                    hidden_parts.append(self._expand_sample_embedding(self.area_emb(area_ids)))
            if self.if_time_in_day:
                tid_ids = self._take_last_index(x_exo_dis, self.tid_idx, self.time_in_day_emb.num_embeddings)
                if tid_ids is not None:
                    hidden_parts.append(self._expand_sample_embedding(self.time_in_day_emb(tid_ids)))
            if self.if_day_in_week:
                diw_ids = self._take_last_index(x_exo_dis, self.diw_idx, self.day_in_week_emb.num_embeddings)
                if diw_ids is not None:
                    hidden_parts.append(self._expand_sample_embedding(self.day_in_week_emb(diw_ids)))

        hidden = torch.cat(hidden_parts, dim=1)
        hidden = self.encoder(hidden)
        prediction = self.regression_layer(hidden)
        return prediction.squeeze(-1)
