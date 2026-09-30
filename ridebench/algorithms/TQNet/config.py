from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class TQNetConfig(ModelConfig):
    """TQNet reproduction metadata.

    Paper: "Temporal Query Network for Efficient Multivariate Time Series
    Forecasting" (ICML 2025).
    Paper URL: https://arxiv.org/pdf/2505.12917
    Source code: https://github.com/ACAT-SCUT/TQNet
    """

    @staticmethod
    def name() -> str:
        return "TQNet"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--use_revin", type=int, choices=[0, 1], default=1, help="use instance norm")
        parser.add_argument("--model_type", type=str, default="mlp", help="model type")
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate")
        parser.add_argument("--d_model", type=int, default=128, help="hidden dim")
        parser.add_argument("--cycle", type=int, default=48 * 7, help="cycle length")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        exo_dis_names = list(exo_dis_dict.keys())
        half_hour_idx = exo_dis_names.index("half_hour_of_day") if "half_hour_of_day" in exo_dis_names else None
        day_of_week_idx = exo_dis_names.index("day_of_week") if "day_of_week" in exo_dis_names else None

        config = Namespace()
        config.cycle = args.cycle
        config.seq_len = input_len
        config.enc_in = x_dim
        config.half_hour_idx = half_hour_idx
        config.pred_len = pred_len
        config.d_model = args.d_model
        config.model_type = args.model_type
        config.use_revin = bool(args.use_revin)
        config.day_of_week_idx = day_of_week_idx
        config.dropout = args.dropout
        return Model(config)
