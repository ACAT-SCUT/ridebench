from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class TimeFilterConfig(ModelConfig):
    """TimeFilter reproduction metadata.

    Paper: "TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for
    Time Series Forecasting" (ICML 2025 Poster).
    Paper URL: https://openreview.net/forum?id=490VcNtjh7
    Source code: https://github.com/TROUBADOUR000/TimeFilter
    """

    @staticmethod
    def name() -> str:
        return "TimeFilter"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--patch_len", type=int, default=48, help="length of patch")
        parser.add_argument(
            "--task_name",
            type=str,
            default="long_term_forecast",
            help="task name, options:[long_term_forecast, short_term_forecast]",
        )
        parser.add_argument("--top_p", type=float, default=0.5, help="dynamic routing in MoE")
        parser.add_argument("--n_heads", type=int, default=4, help="num of heads")
        parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
        parser.add_argument("--d_model", type=int, default=128, help="dimension of model")
        parser.add_argument("--alpha", type=float, default=0.1, help="KNN for graph construction")
        parser.add_argument(
            "--moe_loss_weight",
            type=float,
            default=0.05,
            help="benchmark-side weight for TimeFilter's upstream MoE auxiliary loss",
        )
        parser.add_argument("--pos", type=int, choices=[0, 1], default=1, help="positional embedding flag")
        parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
        parser.add_argument("--d_ff", type=int, default=256, help="dimension of fcn")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        config = Namespace()
        config.alpha = args.alpha
        config.task_name = args.task_name
        config.d_model = args.d_model
        config.top_p = args.top_p
        config.d_ff = args.d_ff
        config.moe_loss_weight = args.moe_loss_weight
        config.pos = bool(args.pos)
        config.n_heads = args.n_heads
        config.patch_len = args.patch_len
        config.c_out = x_dim
        config.dropout = args.dropout
        config.pred_len = pred_len
        config.e_layers = args.e_layers
        config.enc_in = x_dim
        config.seq_len = input_len
        return Model(config)
