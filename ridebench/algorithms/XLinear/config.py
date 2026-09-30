from argparse import ArgumentParser, Namespace

from ..config import ModelConfig
from ..covariate_adapter import add_covariate_arguments, resolve_covariate_spec
from .model import Model


class XLinearConfig(ModelConfig):
    """XLinear reproduction metadata.

    Paper: "XLinear: A Lightweight and Accurate MLP-Based Model for Long-Term
    Time Series Forecasting with Exogenous Inputs" (AAAI 2026).
    Paper URL: https://arxiv.org/abs/2601.09237
    Source code: https://github.com/Zaiwen/XLinear
    Benchmark note: XLinear consumes one dense multivariate tensor. The
    benchmark therefore converts continuous/discrete exogenous variables into
    dense float channels, appends them on the feature axis, and can optionally
    extend the consumed time axis with future-known exogenous inputs plus
    zero-padded endogenous placeholders.
    """

    @staticmethod
    def name() -> str:
        return "XLinear"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--features",
            type=str,
            default="M",
            choices=["M"],
            help="benchmark integration supports multivariate forecasting only",
        )
        parser.add_argument(
            "--usenorm",
            type=int,
            choices=[0, 1],
            default=1,
            help="use instance normalization",
        )
        parser.add_argument("--embed_dropout", type=float, default=0.1, help="input projection dropout")
        parser.add_argument("--head_dropout", type=float, default=0.0, help="forecast head dropout")
        parser.add_argument("--d_model", type=int, default=128, help="model width")
        parser.add_argument("--t_dropout", type=float, default=0.1, help="temporal gating dropout")
        parser.add_argument("--t_ff", type=int, default=256, help="temporal gating hidden size")
        parser.add_argument("--c_dropout", type=float, default=0.1, help="channel gating dropout")
        parser.add_argument("--c_ff", type=int, default=256, help="channel gating hidden size")
        add_covariate_arguments(parser)

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args: Namespace):
        covariates = resolve_covariate_spec(
            x_dim,
            exo_con_list,
            exo_dis_dict,
            args,
            default_use_future_exog=True,
            require_exog_channel=False,
        )

        config = Namespace()
        config.pred_len = pred_len
        config.covariate_spec = covariates
        config.embed_dropout = args.embed_dropout
        config.usenorm = bool(args.usenorm)
        config.d_model = args.d_model
        config.features = args.features
        config.use_future_exog = covariates.use_future_exog
        config.seq_len = input_len
        config.t_dropout = args.t_dropout
        config.target_dim = x_dim
        config.t_ff = args.t_ff
        config.head_dropout = args.head_dropout
        config.enc_in = covariates.enc_in
        config.c_ff = args.c_ff
        config.c_dropout = args.c_dropout
        return Model(config)
