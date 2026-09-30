from argparse import ArgumentParser, Namespace
from ..config import ModelConfig
from ..covariate_adapter import add_covariate_arguments, resolve_covariate_spec
from .model import Model

class DAGConfig(ModelConfig):
    """DAG reproduction metadata.

    Paper: "DAG: A Dual Correlation Network for Time Series Forecasting with
    Exogenous Variables" (current local clone tracks the 2025 preprint).
    Paper URL: https://arxiv.org/pdf/2509.14933
    Source code: https://github.com/decisionintelligence/DAG
    Benchmark note: continuous and discrete exogenous variables are routed
    through the shared covariate adapter; discrete IDs are converted into dense
    features before entering DAG, and future exogenous variables can be exposed
    through the adapter-controlled `use_future_exog` path.
    """

    @staticmethod
    def name() -> str:
        return "DAG"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument('--use_t_exog', action='store_true', default=True, help='allow exogenous features inside the temporal causality branch')
        parser.add_argument('--series_dim', type=int, default=9, help='placeholder endogenous channel count; benchmark resolves the real value from endo_vars')
        parser.add_argument('--alpha', type=float, default=0.5, help='mixture weight assigned to the temporal branch when both branches are enabled')
        parser.add_argument('--use_c_exog', action='store_true', default=True, help='allow exogenous features inside the covariance causality branch')
        parser.add_argument('--activation', type=str, default='gelu', help='encoder activation function')
        parser.add_argument('--d_ff', type=int, default=256, help='feed-forward hidden size inside DAG encoder blocks')
        parser.add_argument('--dropout', type=float, default=0, help='dropout used by DAG attention and projection blocks')
        parser.add_argument('--d_model', type=int, default=128, help='encoder hidden size')
        parser.add_argument('--infer_use_future', action='store_true', default=True, help='legacy DAG flag kept for CLI compatibility; benchmark resolves future exogenous usage from future_exog_mode')
        parser.add_argument('--patch_len', type=int, default=48, choices=[8, 24, 48], help='patch length used by the temporal causality branch')
        parser.add_argument('--beta', type=float, default=0.3, help='scaling factor applied to the auxiliary causality loss')
        parser.add_argument('--criterion', type=int, default=1, help='criterion selector used by the reproduced DAG loss blocks')
        parser.add_argument('--stride', type=int, default=48, choices=[8, 30, 48], help='patch stride used by the temporal causality branch')
        parser.add_argument('--use_c', action='store_true', default=True, help='enable the covariance causality branch')
        parser.add_argument('--use_t', action='store_true', default=True, help='enable the temporal causality branch')
        parser.add_argument('--factor', type=int, default=3, help='attention factor used by FullAttention')
        parser.add_argument('--e_layers', type=int, default=1, help='number of encoder layers per DAG branch')
        parser.add_argument('--n_heads', type=int, default=4, help='number of attention heads')
        parser.add_argument('--enc_in', type=int, default=22, help='placeholder total input channel count; benchmark resolves it from endogenous plus adapted exogenous channels')
        parser.add_argument('--task_name', type=str, default='long_term_forecast',
                            choices=['long_term_forecast', 'short_term_forecast', 'imputation', 'classification', 'anomaly_detection'],
                            help='forecasting task type')
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
        config.criterion = args.criterion
        config.infer_use_future = covariates.use_future_exog
        config.dropout = args.dropout
        config.factor = args.factor
        config.enc_in = covariates.enc_in
        config.e_layers = args.e_layers
        config.pred_len = pred_len
        config.use_c_exog = args.use_c_exog and covariates.use_exog
        config.use_future_exog = covariates.use_future_exog
        config.use_t_exog = args.use_t_exog and covariates.use_exog
        config.beta = args.beta
        config.activation = args.activation
        config.use_c = args.use_c
        config.series_dim = covariates.series_dim
        config.use_t = args.use_t
        config.alpha = args.alpha
        config.covariate_spec = covariates
        config.n_heads = args.n_heads
        config.seq_len = input_len
        config.stride = args.stride
        config.d_ff = args.d_ff
        config.d_model = args.d_model
        config.patch_len = args.patch_len
        config.task_name = args.task_name

        return Model(config)

