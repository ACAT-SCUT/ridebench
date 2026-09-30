from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class LeddamConfig(ModelConfig):
    """Leddam reproduction metadata.

    Paper: "Revitalizing Multivariate Time Series Forecasting: Learnable
    Decomposition with Inter-Series Dependencies and Intra-Series Variations
    Modeling" (ICML 2024).
    Paper URL: https://openreview.net/forum?id=87CYNyCGOo
    Source code: https://github.com/Levi-Ackman/Leddam
    """

    @staticmethod
    def name() -> str:
        return "Leddam"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument(
            "--pe_type",
            type=str,
            default="zeros",
            choices=["no", "zero", "zeros", "normal", "gauss", "uniform", "lin1d", "exp1d", "lin2d", "exp2d", "sincos"],
            help="positional encoding type used by the upstream channel-axis embedding",
        )
        parser.add_argument("--dropout", type=float, default=0.05, help="dropout used by attention projections and feed-forward blocks")
        parser.add_argument("--revin", type=int, choices=[0, 1], default=1, help="enable RevIN normalization before Leddam and inverse normalization after forecasting")
        parser.add_argument("--d_model", type=int, default=128, help="latent feature dimension after channel-wise value embedding")
        parser.add_argument("--kernel_size", type=int, default=25, help="kernel size of the learnable decomposition smoothing filter")
        parser.add_argument("--n_layers", type=int, default=3, help="number of stacked auto-attention and channel-attention residual blocks")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        cfg = Namespace()
        cfg.kernel_size = args.kernel_size
        cfg.seq_len = input_len
        cfg.pe_type = args.pe_type
        cfg.enc_in = x_dim
        cfg.dropout = args.dropout
        cfg.n_layers = args.n_layers
        cfg.pred_len = pred_len
        cfg.revin = bool(args.revin)
        cfg.d_model = args.d_model
        return Model(cfg)
