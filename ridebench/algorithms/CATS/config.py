from argparse import ArgumentParser, Namespace

from ..config import ModelConfig
from ..covariate_adapter import add_covariate_arguments, resolve_covariate_spec
from .model import Model


class CATSConfig(ModelConfig):
    """CATS reproduction metadata.

    Paper: "CATS: Enhancing Multivariate Time Series Forecasting by
    Constructing Auxiliary Time Series as Exogenous Variables" (ICML 2024).
    Paper URL: https://arxiv.org/abs/2403.01673
    Source code: https://github.com/LJC-FVNR/CATS
    Benchmark note: benchmark exogenous variables are adapted into dense float
    channels and appended to the CATS input tensor as auxiliary time series.
    When future-known exogenous inputs are enabled, the benchmark extends the
    sequence with zero-padded future endogenous slots plus known future
    exogenous channels, while the model still predicts endogenous channels only.
    """

    @staticmethod
    def name() -> str:
        return "CATS"

    @staticmethod
    def add_arguments(parser: ArgumentParser) -> None:
        parser.add_argument(
            "--continuity_beta",
            type=float,
            default=1.0,
            help="weight of the tutorial continuity regularizer",
        )
        parser.add_argument(
            "--temporal_gate",
            type=int,
            choices=[0, 1],
            default=1,
            help="enable temporal sparsity gate",
        )
        parser.add_argument(
            "--time_emb_dim",
            type=int,
            default=0,
            help="optional extra timestamp embedding width; benchmark integration leaves this disabled by default",
        )
        parser.add_argument("--F_id_output_rate", type=int, default=1, help="identity ATS expansion rate")
        parser.add_argument("--F_conv_output", type=int, default=4, help="ATS conv branch width")
        parser.add_argument(
            "--predictor",
            type=str,
            default="Default",
            choices=[
                "Default",
                "Simple",
                "MLP",
                "Mean",
                "IndependentLinear",
                "GroupedIndependent",
                "Agg",
            ],
            help="predictor head implemented from the tutorial notebook",
        )
        parser.add_argument("--F_lin_output", type=int, default=2, help="ATS 1x1 projection branch width")
        parser.add_argument("--F_gconv_output_rate", type=int, default=1, help="grouped conv expansion rate")
        parser.add_argument(
            "--label_len",
            type=int,
            default=0,
            help="decoder warmup length; 0 keeps predictor-specific decoder inputs disabled",
        )
        parser.add_argument("--mlp_ratio", type=int, default=4, help="shared hidden expansion ratio")
        parser.add_argument("--F_emb_output", type=int, default=2, help="learned ATS embedding width")
        parser.add_argument("--predictor_dropout", type=float, default=0.0, help="predictor dropout")
        parser.add_argument("--F_noconv_output", type=int, default=4, help="ATS non-overlapping conv branch width")
        parser.add_argument(
            "--channel_sparsity",
            type=int,
            choices=[0, 1],
            default=1,
            help="enable channel sparsity gating",
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
            require_exog_channel=False,
        )
        config = Namespace()
        uses_future_exog_input = covariates.use_future_exog and covariates.model_exog_dim > 0
        config.F_conv_output = args.F_conv_output
        config.F_gconv_output_rate = args.F_gconv_output_rate
        config.features = "M"
        config.F_noconv_output = args.F_noconv_output
        config.pred_len = pred_len
        config.enc_in = covariates.enc_in
        config.channel_sparsity = bool(args.channel_sparsity)
        config.F_id_output_rate = args.F_id_output_rate
        config.target_dim = x_dim
        config.predictor_dropout = args.predictor_dropout
        config.time_emb_dim = args.time_emb_dim
        config.covariate_spec = covariates
        config.F_emb_output = args.F_emb_output
        config.F_lin_output = args.F_lin_output
        config.number_of_targets = x_dim
        config.mlp_ratio = args.mlp_ratio
        config.predictor = args.predictor
        config.continuity_beta = args.continuity_beta
        config.seq_len = input_len + pred_len if uses_future_exog_input else input_len
        config.temporal_gate = bool(args.temporal_gate)
        config.label_len = args.label_len
        return Model(config)
