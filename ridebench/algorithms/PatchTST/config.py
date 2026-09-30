from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class PatchTSTConfig(ModelConfig):
    """PatchTST reproduction metadata.

    Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with
    Transformers" (ICLR 2023).
    Paper URL: https://arxiv.org/abs/2211.14730
    Source code: https://github.com/yuqinie98/PatchTST
    """

    @staticmethod
    def name() -> str:
        return "PatchTST"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--d_model", type=int, default=128, help="transformer d_model")
        parser.add_argument("--d_ff", type=int, default=256, help="feedforward dim")
        parser.add_argument("--stride", type=int, default=48, help="patch stride")
        parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads")
        parser.add_argument("--revin", type=int, choices=[0, 1], default=1, help="1: enable simple revIN-like norm")
        parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
        parser.add_argument("--e_layers", type=int, default=3, help="number of encoder layers")
        parser.add_argument("--head_dropout", type=float, default=0.0, help="dropout for head")
        parser.add_argument("--patch_len", type=int, default=48, help="patch length")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        cfg = Namespace()
        cfg.head_dropout = args.head_dropout
        cfg.enc_in = x_dim
        cfg.d_model = args.d_model

        cfg.pred_len = pred_len
        cfg.n_heads = args.n_heads
        cfg.dropout = args.dropout
        cfg.d_ff = args.d_ff
        cfg.e_layers = args.e_layers
        cfg.stride = args.stride
        cfg.seq_len = input_len
        cfg.revin = args.revin
        cfg.patch_len = args.patch_len

        return Model(cfg)
