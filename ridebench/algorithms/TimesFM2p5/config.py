import os

from ..config import ModelConfig
from .model import Model


class TimesFM2p5Config(ModelConfig):
    """TimesFM 2.5 reproduction metadata.

    Paper: "A decoder-only foundation model for time-series forecasting"
    (ICML 2024; this wrapper uses the newer TimesFM 2.5 checkpoint family).
    Paper URL: https://arxiv.org/abs/2310.10688
    Source code: https://github.com/google-research/timesfm
    Benchmark note: this local integration evaluates TimesFM 2.5 as a
    per-target forecaster in endogenous-only mode.
    """

    @staticmethod
    def name() -> str:
        return "TimesFM2p5"

    @staticmethod
    def need_train() -> bool:
        return False

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument(
            "--timesfm_local_files_only",
            type=int,
            default=0,
            help="Restrict Hugging Face fallback loading to local files only (1=True, 0=False).",
        )
        parser.add_argument(
            "--timesfm_max_horizon",
            type=int,
            default=0,
            help="Maximum horizon passed to TimesFM compile; 0 uses pred_len.",
        )
        parser.add_argument(
            "--timesfm_torch_compile",
            type=int,
            default=0,
            help="Compile the upstream TimesFM torch module on load (1=True, 0=False).",
        )
        parser.add_argument(
            "--timesfm_fix_quantile_crossing",
            type=int,
            default=1,
            help="Fix TimesFM quantile crossing (1=True, 0=False).",
        )
        parser.add_argument(
            "--timesfm_infer_is_positive",
            type=int,
            default=0,
            help="Infer non-negative outputs when the input is non-negative (1=True, 0=False).",
        )
        parser.add_argument(
            "--timesfm_max_context",
            type=int,
            default=0,
            help="Maximum context passed to TimesFM compile; 0 uses input_len.",
        )
        parser.add_argument(
            "--timesfm_force_flip_invariance",
            type=int,
            default=1,
            help="Enable TimesFM flip invariance (1=True, 0=False).",
        )
        parser.add_argument(
            "--timesfm_normalize_inputs",
            type=int,
            default=1,
            help="Normalize inputs before TimesFM decoding (1=True, 0=False).",
        )
        parser.add_argument(
            "--timesfm_model_path",
            type=str,
            default=os.environ.get("BENCH_TIMESFM2P5_MODEL", "./pretrained_models/timesfm-2.5-200m-pytorch"),
            help="TimesFM 2.5 model id or local checkpoint directory/file.",
        )
        parser.add_argument(
            "--timesfm_per_core_batch_size",
            type=int,
            default=128,
            help="TimesFM per-core batch size used during compiled decoding.",
        )
        parser.add_argument(
            "--timesfm_device",
            type=str,
            default="auto",
            choices=["auto", "cpu", "cuda"],
            help="Runtime device for the local torch TimesFM wrapper.",
        )
        parser.add_argument(
            "--timesfm_use_continuous_quantile_head",
            type=int,
            default=0,
            help="Enable TimesFM continuous quantile head (1=True, 0=False).",
        )
    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        del exo_con_list, exo_dis_dict
        return Model(timesfm_infer_is_positive=bool(args.timesfm_infer_is_positive), pred_len=pred_len, timesfm_max_horizon=args.timesfm_max_horizon, timesfm_max_context=args.timesfm_max_context, timesfm_model_path=args.timesfm_model_path, input_len=input_len, timesfm_per_core_batch_size=args.timesfm_per_core_batch_size, timesfm_normalize_inputs=bool(args.timesfm_normalize_inputs), timesfm_fix_quantile_crossing=bool(args.timesfm_fix_quantile_crossing), timesfm_local_files_only=bool(args.timesfm_local_files_only), timesfm_force_flip_invariance=bool(args.timesfm_force_flip_invariance), timesfm_use_continuous_quantile_head=bool(args.timesfm_use_continuous_quantile_head), timesfm_torch_compile=bool(args.timesfm_torch_compile), timesfm_device=args.timesfm_device)
