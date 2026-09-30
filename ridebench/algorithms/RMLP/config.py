from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class RMLPConfig(ModelConfig):
    """RMLP reproduction metadata.

    Paper: "Revisiting Long-term Time Series Forecasting: An Investigation on
    Linear Mapping" (2023).
    Paper URL: https://arxiv.org/abs/2305.10721
    Source code: https://github.com/plumprc/RTSF
    """

    @staticmethod
    def name() -> str:
        return "RMLP"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--rev", type=int, default=1, help="apply RevIN normalization and de-normalization")
        parser.add_argument("--d_model", type=int, default=128, help="hidden size of the residual temporal MLP")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        del exo_con_list, exo_dis_dict
        cfg = Namespace()
        cfg.pred_len = pred_len
        cfg.rev = bool(args.rev)
        cfg.channel = x_dim
        cfg.seq_len = input_len
        cfg.d_model = args.d_model
        return Model(cfg)
