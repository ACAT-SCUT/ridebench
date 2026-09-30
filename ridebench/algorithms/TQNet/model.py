import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.use_revin = configs.use_revin
        self.half_hour_idx = configs.half_hour_idx
        self.day_of_week_idx = configs.day_of_week_idx

        self.use_tq = True
        self.channel_aggre = True

        if self.use_tq:
            self.temporalQuery = nn.Parameter(torch.zeros(self.cycle_len, self.enc_in), requires_grad=True)

        if self.channel_aggre:
            self.channelAggregator = nn.MultiheadAttention(
                embed_dim=self.seq_len,
                num_heads=4,
                batch_first=True,
                dropout=0.5,
            )

        self.input_proj = nn.Linear(self.seq_len, self.d_model)
        self.model = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
        )
        self.output_proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.pred_len),
        )

    def _cycle_index_from_exo(self, x, x_exo_dis, y_exo_dis):
        reference_exo_dis = y_exo_dis if y_exo_dis is not None else x_exo_dis
        if reference_exo_dis is None or self.half_hour_idx is None or self.day_of_week_idx is None:
            return torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        if reference_exo_dis.shape[-1] <= max(self.half_hour_idx, self.day_of_week_idx):
            return torch.zeros(x.shape[0], device=x.device, dtype=torch.long)

        # Benchmark compatibility: for the default 30-minute ride-hailing data,
        # TQNet's temporal query index is the weekly slot index inside a 48*7 cycle.
        half_hour = reference_exo_dis[:, 0, self.half_hour_idx].long()
        day_of_week = reference_exo_dis[:, 0, self.day_of_week_idx].long()
        return (day_of_week * 48 + half_hour) % self.cycle_len

    def forward(self, x, x_exo_con=None, x_exo_dis=None, y_exo_con=None, y_exo_dis=None):
        cycle_index = self._cycle_index_from_exo(x, x_exo_dis, y_exo_dis)

        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_input = x.permute(0, 2, 1)

        if self.use_tq:
            gather_index = (cycle_index.view(-1, 1) + torch.arange(self.seq_len, device=cycle_index.device).view(1, -1)) % self.cycle_len
            query_input = self.temporalQuery[gather_index].permute(0, 2, 1)
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=query_input, key=x_input, value=x_input)[0]
            else:
                channel_information = query_input
        else:
            if self.channel_aggre:
                channel_information = self.channelAggregator(query=x_input, key=x_input, value=x_input)[0]
            else:
                channel_information = 0

        model_input = self.input_proj(x_input + channel_information)
        hidden = self.model(model_input)
        output = self.output_proj(hidden + model_input).permute(0, 2, 1)

        if self.use_revin:
            output = output * torch.sqrt(seq_var) + seq_mean

        return output
