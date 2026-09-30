from argparse import ArgumentParser, Namespace

from ..config import ModelConfig
from ..covariate_adapter import add_covariate_arguments, resolve_covariate_spec
from .model import Model


class TiDEConfig(ModelConfig):
    """TiDE reproduction metadata.

    Paper: "Long-term Forecasting with TiDE: Time-series Dense Encoder"
    (TMLR 2023).
    Paper URL: https://arxiv.org/pdf/2304.08424.pdf
    Source code: benchmark implementation follows the TiDE baseline published
    in TSLib-style model collections and uses the TiDE copy already integrated
    in the DAG codebase as the direct implementation base.
    Benchmark note: benchmark covariates are injected through the shared
    adapter, which turns continuous/discrete exogenous inputs into dense
    features and optionally exposes future known covariates.
    """

    @staticmethod
    def name() -> str:
        return "TiDE"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument("--e_layers", type=int, default=1, help="encoder block count")
        parser.add_argument("--freq", type=str, default="h", help="time feature frequency flag used by the reproduced interface")
        parser.add_argument(
            "--task_name",
            type=str,
            default="long_term_forecast",
            choices=[
                "long_term_forecast",
                "short_term_forecast",
                "imputation",
                "classification",
                "anomaly_detection",
            ],
            help="forecasting task type exposed by the reproduced TiDE CLI",
        )
        parser.add_argument("--label_len", type=int, default=48, help="decoder warmup length")
        parser.add_argument("--d_model", type=int, default=128, help="model width")
        parser.add_argument("--d_layers", type=int, default=2, help="decoder block count")
        parser.add_argument("--d_ff", type=int, default=256, help="temporal decoder hidden size")
        parser.add_argument("--dropout", type=float, default=0.1, help="dropout used by TiDE encoder/decoder blocks")
        parser.add_argument(
            "--feature_encode_dim",
            type=int,
            default=2,
            help="encoded feature dimension for each exogenous channel",
        )
        parser.add_argument(
            "--use_future_exog",
            action="store_true",
            default=True,
            help="enable future-known exogenous variables in the benchmark adapter",
        )
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
        config.d_layers = args.d_layers
        config.feature_encode_dim = args.feature_encode_dim
        config.pred_len = pred_len
        config.seq_len = input_len
        config.c_out = x_dim
        config.dropout = args.dropout
        config.freq = args.freq
        config.covariate_spec = covariates
        config.use_future_exog = covariates.use_future_exog
        config.d_ff = args.d_ff
        config.d_model = args.d_model
        config.task_name = args.task_name
        config.label_len = args.label_len
        config.covariate_dim = covariates.model_exog_dim
        config.e_layers = args.e_layers
        return Model(config)
