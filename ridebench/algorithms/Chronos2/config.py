import os

from ..config import ModelConfig
from .model import Model


class Chronos2Config(ModelConfig):
    """Chronos-2 reproduction metadata.

    Paper: "Chronos-2: From Univariate to Universal Forecasting" (2025
    release).
    Paper URL: https://arxiv.org/abs/2510.15821
    Source code: https://github.com/amazon-science/chronos-forecasting
    Benchmark note: this wrapper keeps the benchmark's multivariate endogenous
    task intact and passes continuous/discrete exogenous variables through the
    Chronos-2 covariate interface without a local training loop.
    """

    @staticmethod
    def name() -> str:
        return "Chronos2"

    @staticmethod
    def need_train() -> bool:
        return False

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument(
            "--chronos_local_files_only",
            type=int,
            default=0,
            help="Restrict model loading to local files only (1=True, 0=False).",
        )
        parser.add_argument(
            "--use_exog",
            type=int,
            default=1,
            help="Use benchmark exogenous covariates (1=True, 0=False).",
        )
        parser.add_argument(
            "--chronos_batch_size",
            type=int,
            default=32,
            help="Forecast batch size used by Chronos-2 pipeline.",
        )
        parser.add_argument(
            "--chronos_model_path",
            type=str,
            default=os.environ.get("BENCH_CHRONOS2_MODEL", "./pretrained_models/chronos-2"),
            help="Chronos-2 model id or local checkpoint directory.",
        )
        parser.add_argument(
            "--chronos_torch_dtype",
            type=str,
            default="auto",
            choices=["auto", "float32", "float16", "bfloat16"],
            help="Torch dtype passed to Chronos-2 loading.",
        )
        parser.add_argument(
            "--chronos_context_length",
            type=int,
            default=0,
            help="Optional context length cap; 0 uses the model default.",
        )
        parser.add_argument(
            "--chronos_cross_learning",
            type=int,
            default=0,
            help="Enable Chronos-2 cross-learning across benchmark windows (1=True, 0=False).",
        )
        parser.add_argument(
            "--chronos_device_map",
            type=str,
            default="auto",
            choices=["auto", "cpu", "cuda"],
            help="Device map passed to Chronos-2 loading.",
        )

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        return Model(chronos_batch_size=args.chronos_batch_size, chronos_context_length=args.chronos_context_length, chronos_device_map=args.chronos_device_map, exo_con_names=list(exo_con_list), exo_dis_names=list(exo_dis_dict.keys()), chronos_torch_dtype=args.chronos_torch_dtype, chronos_model_path=args.chronos_model_path, use_exog=bool(args.use_exog), chronos_cross_learning=bool(args.chronos_cross_learning), chronos_local_files_only=bool(args.chronos_local_files_only), pred_len=pred_len)
