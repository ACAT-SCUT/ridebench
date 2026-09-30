from argparse import ArgumentParser, Namespace

from ..config import ModelConfig
from ..covariate_adapter import add_covariate_arguments, resolve_covariate_spec
from .model import Model


class TimeXerConfig(ModelConfig):
    """TimeXer reproduction metadata.

    Paper: "TimeXer: Empowering Transformers for Time Series Forecasting with
    Exogenous Variables" (2024 release).
    Paper URL: https://arxiv.org/abs/2402.19072
    Source code: https://github.com/thuml/TimeXer
    Benchmark note: benchmark discrete exogenous variables are densified by the
    shared adapter before entering TimeXer. The local benchmark defaults to
    using future-known exogenous inputs whenever exogenous features are
    enabled, by concatenating future covariates to the exogenous
    cross-attention context.
    """

    @staticmethod
    def name() -> str:
        return "TimeXer"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument('--series_dim', type=int, default=9, help='placeholder endogenous channel count; benchmark resolves the real value from endo_vars')
        parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers')
        parser.add_argument('--n_heads', type=int, default=4, help='number of attention heads')
        parser.add_argument('--activation', type=str, default='gelu', help='encoder activation function')
        parser.add_argument('--enc_in', type=int, default=22, help='placeholder total input channel count; benchmark resolves it from endogenous plus adapted exogenous channels')
        parser.add_argument('--factor', type=int, default=3, help='attention factor used by FullAttention')
        parser.add_argument('--stride', type=int, default=48, choices=[8, 24, 48], help='patch stride used by the upstream TimeXer patcher')
        parser.add_argument('--d_ff', type=int, default=256, help='feed-forward hidden size inside encoder layers')
        parser.add_argument('--patch_len', type=int, default=48, choices=[8, 24, 48], help='patch length used for endogenous history segmentation')
        parser.add_argument('--task_name', type=str, default='long_term_forecast',
                            choices=['long_term_forecast', 'short_term_forecast', 'imputation', 'classification', 'anomaly_detection'],
                            help='forecasting task type')
        parser.add_argument('--freq', type=str, default='h', help='time feature frequency flag used by the exogenous embedding path')
        parser.add_argument('--d_model', type=int, default=128, help='transformer hidden size')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout applied in embeddings, attention, and forecast head')
        add_covariate_arguments(parser)


    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        covariates = resolve_covariate_spec(
            x_dim,
            exo_con_list,
            exo_dis_dict,
            args,
            default_use_future_exog=True,
            require_exog_channel=True,
        )
        config = Namespace()
        config.dropout = args.dropout
        config.patch_len = args.patch_len
        config.e_layers = args.e_layers
        config.n_heads = args.n_heads
        config.d_ff = args.d_ff
        config.use_future_exog = covariates.use_future_exog
        config.factor = args.factor
        config.seq_len = input_len
        config.exog_len = input_len + (pred_len if covariates.use_future_exog else 0)
        config.pred_len = pred_len
        config.activation = args.activation
        config.covariate_spec = covariates
        config.freq = args.freq
        config.task_name = args.task_name
        config.enc_in = covariates.enc_in
        config.d_model = args.d_model
        config.series_dim = covariates.series_dim

        return Model(config)
