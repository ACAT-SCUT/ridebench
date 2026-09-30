from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class TimeBaseConfig(ModelConfig):
    """TimeBase reproduction metadata.

    Paper: "TimeBase: The Power of Minimalism in Efficient Long-term Time
    Series Forecasting" (ICML 2025 Spotlight).
    Paper URL: https://proceedings.mlr.press/v267/huang25az.html
    Source code: https://github.com/hqh0728/TimeBase
    Benchmark note: local defaults use the benchmark's half-hour periodic prior
    (`period_len=48`) instead of the shorter periods often used upstream.
    """

    @staticmethod
    def name() -> str:
        return "TimeBase"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--individual", type=int, choices=[0, 1], default=0, help="individual channel head flag")
        parser.add_argument("--use_period_norm", type=int, choices=[0, 1], default=1, help="period normalization flag")
        parser.add_argument("--period_len", type=int, default=48, help="period length")
        parser.add_argument("--basis_num", type=int, default=6, help="basis number")
        parser.add_argument("--use_orthogonal", type=int, choices=[0, 1], default=0, help="orthogonal regularization flag")
        parser.add_argument("--orthogonal_weight", type=float, default=0.2, help="orthogonal regularization weight")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        config = Namespace()
        config.period_len = args.period_len
        config.basis_num = args.basis_num
        config.individual = bool(args.individual)
        config.pred_len = pred_len
        config.seq_len = input_len
        config.use_orthogonal = bool(args.use_orthogonal)
        config.enc_in = x_dim
        config.use_period_norm = bool(args.use_period_norm)
        return Model(config)
