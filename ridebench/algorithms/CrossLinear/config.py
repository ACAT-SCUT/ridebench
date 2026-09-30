from argparse import Namespace

from ..config import ModelConfig
from ..covariate_adapter import add_covariate_arguments, resolve_covariate_spec
from .model import Model


class CrossLinearConfig(ModelConfig):
    """CrossLinear reproduction metadata.

    Paper: "CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time
    Series Forecasting with Exogenous Variables" (KDD 2025).
    Paper URL: https://arxiv.org/pdf/2505.23116
    Source code: https://github.com/mumiao2000/CrossLinear
    Benchmark note: upstream CrossLinear consumes one multivariate tensor only.
    The benchmark therefore adapts exogenous variables into dense float
    channels, appends historical exogenous inputs on the feature axis, and can
    optionally extend the consumed history length with future-known exogenous
    inputs plus zero-padded endogenous placeholders.
    """

    @staticmethod
    def name() -> str:
        return "CrossLinear"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument(
            "--features",
            type=str,
            default="M",
            choices=["M", "S", "MS"],
            help="forecasting task, options:[M, S, MS]",
        )
        parser.add_argument("--d_model", type=int, default=128, help="dimension of model")
        parser.add_argument("--d_ff", type=int, default=256, help="dimension of fcn")
        parser.add_argument("--patch_len", type=int, default=48, help="patch length")
        parser.add_argument("--beta", type=float, default=1.0, help="beta")
        parser.add_argument(
            "--task_name",
            type=str,
            default="long_term_forecast",
            help="task name, options:[long_term_forecast, short_term_forecast]",
        )
        parser.add_argument("--alpha", type=float, default=1.0, help="alpha")
        add_covariate_arguments(parser)

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        covariates = resolve_covariate_spec(
            x_dim,
            exo_con_list,
            exo_dis_dict,
            args,
            default_use_future_exog=True,
            require_exog_channel=False,
        )
        use_future_exog = covariates.use_future_exog and covariates.model_exog_dim > 0

        config = Namespace()
        config.d_model = args.d_model
        config.use_future_exog = use_future_exog
        config.features = args.features
        config.d_ff = args.d_ff
        config.beta = args.beta
        config.target_dim = x_dim
        config.dec_in = covariates.enc_in
        config.task_name = args.task_name
        config.covariate_spec = covariates
        config.pred_len = pred_len
        config.seq_len = input_len + (pred_len if use_future_exog else 0)
        config.patch_len = args.patch_len
        config.alpha = args.alpha
        return Model(config)
