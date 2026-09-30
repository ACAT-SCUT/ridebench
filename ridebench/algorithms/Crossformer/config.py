from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class CrossformerConfig(ModelConfig):
    """Crossformer reproduction metadata.

    Paper: "Crossformer: Transformer Utilizing Cross-Dimension Dependency for
    Multivariate Time Series Forecasting" (ICLR 2023).
    Paper URL: https://openreview.net/forum?id=vSVLM2j9eie
    Source code: https://github.com/Thinklab-SJTU/Crossformer
    """

    @staticmethod
    def name() -> str:
        return "Crossformer"

    @staticmethod
    def add_arguments(parser) -> None:
        # Benchmark adaptation: half-hourly data has a strong daily cycle, so
        # one segment covers one day and keeps regular/ultra runs tractable.
        parser.add_argument("--factor", type=int, default=10, help="router count in cross-dimension attention")
        parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
        parser.add_argument("--d_ff", type=int, default=256, help="dimension of transformer MLP")
        parser.add_argument("--baseline", action="store_true", help="use the mean of the input sequence as prediction baseline")
        parser.add_argument("--seg_len", type=int, default=48, help="segment length")
        parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads")
        parser.add_argument("--e_layers", type=int, default=3, help="number of encoder layers")
        parser.add_argument("--d_model", type=int, default=128, help="dimension of hidden states")
        parser.add_argument("--win_size", type=int, default=4, help="segment merge window size")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        cfg = Namespace()
        cfg.dropout = args.dropout
        cfg.data_dim = x_dim
        cfg.factor = args.factor
        cfg.in_len = input_len
        cfg.e_layers = args.e_layers
        cfg.baseline = args.baseline
        cfg.n_heads = args.n_heads
        cfg.seg_len = args.seg_len
        cfg.win_size = args.win_size
        cfg.d_ff = args.d_ff
        cfg.d_model = args.d_model
        cfg.out_len = pred_len
        return Model(cfg)
