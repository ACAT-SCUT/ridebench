from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class PhaseFormerConfig(ModelConfig):
    """PhaseFormer reproduction metadata.

    Paper: "PhaseFormer: From Patches to Phases for Efficient and Effective
    Time Series Forecasting" (ICLR 2026).
    Paper URL: https://arxiv.org/abs/2510.04134
    Source code: https://github.com/neumyor/PhaseFormer_TSL
    """

    @staticmethod
    def name() -> str:
        return "PhaseFormer"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument('--phase_use_pos_embed', type=int, default=1, help='use phase positional embedding (1/0)')
        parser.add_argument('--phase_layers', type=int, default=1, help='cross-phase routing layers')
        parser.add_argument('--task_name', type=str, default='long_term_forecast',
                            choices=['long_term_forecast', 'short_term_forecast', 'imputation', 'classification', 'anomaly_detection'],
                            help='task name')
        parser.add_argument('--phase_attn_dropout', type=float, default=0.1, help='phase attention dropout')
        parser.add_argument('--phase_encoder_dropout', type=float, default=0.0, help='phase encoder dropout')
        parser.add_argument('--predictor_use_mlp', type=int, default=0, help='use mlp in predictor (1/0)')
        parser.add_argument('--phase_encoder_use_mlp', type=int, default=0, help='use mlp in phase encoder (1/0)')
        parser.add_argument('--revin_eps', type=float, default=1e-5, help='RevIN epsilon')
        parser.add_argument('--period_len', type=int, default=48, help='period length')
        parser.add_argument('--phase_attn_heads', type=int, default=4, help='phase attention heads')
        parser.add_argument('--latent_dim', type=int, default=8, help='latent dimension')
        parser.add_argument('--revin_affine', type=int, default=0, help='RevIN affine (1/0)')
        parser.add_argument('--predictor_dropout', type=float, default=0.0, help='predictor dropout')
        parser.add_argument('--phase_pos_dropout', type=float, default=0.0, help='phase positional dropout')
        parser.add_argument('--phase_encoder_hidden', type=int, default=32, help='phase encoder hidden size')
        parser.add_argument('--predictor_hidden', type=int, default=64, help='predictor hidden size')
        parser.add_argument('--use_revin', type=int, default=1, help='use RevIN (1/0)')
        parser.add_argument('--phase_num_routers', type=int, default=8, help='number of routing tokens')

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        config = Namespace()
        config.period_len = args.period_len
        config.predictor_dropout = args.predictor_dropout
        config.phase_use_pos_embed = bool(args.phase_use_pos_embed)
        config.phase_encoder_use_mlp = bool(args.phase_encoder_use_mlp)

        config.phase_attn_dropout = args.phase_attn_dropout
        config.task_name = args.task_name
        config.revin_eps = args.revin_eps
        config.seq_len = input_len

        config.latent_dim = args.latent_dim
        config.enc_in = x_dim
        config.predictor_use_mlp = bool(args.predictor_use_mlp)
        config.phase_encoder_dropout = args.phase_encoder_dropout
        config.use_revin = bool(args.use_revin)
        config.phase_encoder_hidden = args.phase_encoder_hidden

        config.pred_len = pred_len
        config.phase_layers = args.phase_layers
        config.revin_affine = bool(args.revin_affine)

        config.predictor_hidden = args.predictor_hidden
        config.phase_attn_heads = args.phase_attn_heads
        config.phase_pos_dropout = args.phase_pos_dropout
        config.phase_num_routers = args.phase_num_routers

        return Model(config)
