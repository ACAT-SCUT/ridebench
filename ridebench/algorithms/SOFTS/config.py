from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class SOFTSConfig(ModelConfig):
    """SOFTS reproduction metadata.

    Paper: "SOFTS: Efficient Multivariate Time Series Forecasting with
    Series-Core Fusion" (NeurIPS 2024).
    Paper URL: https://arxiv.org/abs/2404.14197
    Source code: https://github.com/Secilia-Cxy/SOFTS
    """

    @staticmethod
    def name() -> str:
        return "SOFTS"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--d_model", type=int, default=128, help="dimension of model")
        parser.add_argument("--activation", type=str, default="gelu", help="activation")
        parser.add_argument("--d_ff", type=int, default=256, help="dimension of fcn")
        parser.add_argument("--use_norm", type=int, default=1, help="use normalization and denormalization")
        parser.add_argument("--d_core", type=int, default=128, help="dimension of series core")
        parser.add_argument("--e_layers", type=int, default=2, help="number of encoder layers")
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        cfg = Namespace()
        cfg.use_norm = args.use_norm
        cfg.activation = args.activation
        cfg.e_layers = args.e_layers
        cfg.d_core = args.d_core
        cfg.dropout = args.dropout
        cfg.d_model = args.d_model
        cfg.pred_len = pred_len
        cfg.d_ff = args.d_ff
        cfg.seq_len = input_len
        return Model(cfg)
