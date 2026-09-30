from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class PMDformerConfig(ModelConfig):
    """PMDformer reproduction metadata.

    Paper: "PMDformer: Patch-Mean Decoupling Transformer for Long-term
    Forecasting" (ICLR 2026 Poster).
    Paper URL: https://openreview.net/forum?id=rfJ41gK9Ct
    Source code: https://github.com/aohu1105/PMDformer
    """

    @staticmethod
    def name() -> str:
        return "PMDformer"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--n_heads", type=int, default=8, help="number of attention heads")
        parser.add_argument("--use_norm", type=int, default=1, help="whether to use norm and denorm")
        parser.add_argument("--d_ff", type=int, default=256, help="dimension of feed-forward layer")
        parser.add_argument("--d_model", type=int, default=128, help="dimension of model")
        parser.add_argument("--patch_size", type=int, default=48, help="patch size")
        parser.add_argument("--v_layers", type=int, default=2, help="number of PVA encoder layers")
        parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
        parser.add_argument("--e_layers", type=int, default=2, help="number of TRA encoder layers")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        config = Namespace()
        config.enc_in = x_dim
        config.seq_len = input_len
        config.n_heads = args.n_heads
        config.dropout = args.dropout
        config.d_ff = args.d_ff
        config.pred_len = pred_len
        config.v_layers = args.v_layers
        config.use_norm = args.use_norm
        config.patch_size = args.patch_size
        config.d_model = args.d_model
        config.e_layers = args.e_layers
        return Model(config)
