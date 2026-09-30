from argparse import ArgumentParser, Namespace

from ..config import ModelConfig
from .model import Model


class PETformerConfig(ModelConfig):
    """PETformer reproduction metadata.

    Paper: "PETformer: Long-term Time Series Forecasting via
    Placeholder-enhanced Transformer" (IEEE TETCI 2025).
    Paper URL: https://arxiv.org/pdf/2308.04791
    Source code: https://github.com/ACAT-SCUT/PETformer
    """

    @staticmethod
    def name() -> str:
        return "PETformer"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument("--attn_type", type=int, default=0, choices=[0, 1, 2, 3], help="attention mask type")
        parser.add_argument("--win_stride", type=int, default=48, help="sub-sequence stride")
        parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
        parser.add_argument("--win_len", type=int, default=48, help="sub-sequence window length")
        parser.add_argument("--ff_factor", type=int, default=2, help="feed-forward multiplier")
        parser.add_argument("--channel_attn", type=int, default=0, choices=[0, 1, 2, 3], help="channel interaction mode")
        parser.add_argument("--e_layers", type=int, default=3, help="number of transformer encoder layers")
        parser.add_argument("--d_model", type=int, default=128, help="transformer hidden size")
        parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        config = Namespace()
        config.win_len = args.win_len
        config.dropout = args.dropout
        config.attn_type = args.attn_type
        config.d_model = args.d_model
        config.enc_in = x_dim
        config.channel_attn = args.channel_attn
        config.pred_len = pred_len
        config.ff_factor = args.ff_factor
        config.n_heads = args.n_heads
        config.win_stride = args.win_stride
        config.seq_len = input_len
        config.e_layers = args.e_layers
        return Model(config)
