from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class DUETConfig(ModelConfig):
    """DUET reproduction metadata.

    Paper: "DUET: Dual Clustering Enhanced Multivariate Time Series
    Forecasting" (KDD 2025).
    Paper URL: https://arxiv.org/pdf/2412.10859
    Source code: https://github.com/decisionintelligence/DUET
    """

    @staticmethod
    def name() -> str:
        return "DUET"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--n_heads", type=int, default=4, help="number of channel-attention heads")
        parser.add_argument("--d_model", type=int, default=128, help="DUET temporal feature dimension")
        parser.add_argument("--CI", type=int, default=1, help="enable channel-independent temporal clustering")
        parser.add_argument("--activation", type=str, default="gelu", help="encoder activation")
        parser.add_argument("--dropout", type=float, default=0.2, help="attention dropout")
        parser.add_argument("--e_layers", type=int, default=2, help="number of channel-attention encoder layers")
        parser.add_argument("--factor", type=int, default=3, help="attention factor")
        parser.add_argument("--noisy_gating", type=int, default=1, help="use noisy top-k gating during training")
        parser.add_argument("--fc_dropout", type=float, default=0.05, help="prediction head dropout")
        parser.add_argument("--d_ff", type=int, default=256, help="transformer feed-forward dimension")
        parser.add_argument("--k", type=int, default=2, help="top-k experts selected by the router")
        parser.add_argument("--hidden_size", type=int, default=256, help="router hidden dimension")
        parser.add_argument("--num_experts", type=int, default=4, help="number of temporal experts")
        parser.add_argument("--moving_avg", type=int, default=25, help="decomposition moving-average kernel size")
        parser.add_argument("--output_attention", action="store_true", help="return attention weights inside DUET attention blocks")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        cfg = Namespace()
        cfg.seq_len = input_len
        cfg.k = args.k
        cfg.fc_dropout = args.fc_dropout
        cfg.dropout = args.dropout
        cfg.num_experts = args.num_experts
        cfg.factor = args.factor
        cfg.pred_len = pred_len
        cfg.noisy_gating = bool(args.noisy_gating)
        cfg.d_model = args.d_model
        cfg.CI = bool(args.CI)
        cfg.output_attention = args.output_attention
        cfg.activation = args.activation
        cfg.moving_avg = args.moving_avg
        cfg.d_ff = args.d_ff
        cfg.enc_in = x_dim
        cfg.hidden_size = args.hidden_size
        cfg.e_layers = args.e_layers
        cfg.n_heads = args.n_heads
        return Model(cfg)
