import os

from ..config import ModelConfig
from .model import Model


class TimerS1Config(ModelConfig):
    """Timer-S1 reproduction metadata.

    Paper: "Timer-S1: A Billion-Scale Time Series Foundation Model with
    Serial Scaling" (2026 release).
    Paper URL: https://arxiv.org/abs/2603.04791
    Source page: https://huggingface.co/bytedance-research/Timer-S1
    Benchmark note: Timer-S1 is integrated as an endogenous-only per-target
    foundation model. The benchmark reshapes each endogenous channel into an
    independent zero-shot task and uses the 0.5 quantile as the point forecast.
    """

    @staticmethod
    def name() -> str:
        return "TimerS1"

    @staticmethod
    def need_train() -> bool:
        return False

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument(
            "--timers1_torch_dtype",
            type=str,
            default="float32",
            choices=["auto", "float32", "float16", "bfloat16"],
            help="Torch dtype passed to Timer-S1 loading.",
        )
        parser.add_argument(
            "--timers1_use_cache",
            type=int,
            default=0,
            help="Keep Timer-S1 KV cache enabled during generation (1=True, 0=False).",
        )
        parser.add_argument(
            "--timers1_model_path",
            type=str,
            default=os.environ.get("BENCH_TIMERS1_MODEL", "./pretrained_models/Timer-S1"),
            help="Timer-S1 local checkpoint directory or Hugging Face model id.",
        )
        parser.add_argument(
            "--use_exog",
            type=int,
            default=0,
            help="Unused for Timer-S1. Kept at 0 because the current benchmark integration follows the official endogenous-only API.",
        )
        parser.add_argument(
            "--timers1_context_length",
            type=int,
            default=0,
            help="Optional context cap; 0 keeps the full benchmark input window subject to model limits.",
        )
        parser.add_argument(
            "--timers1_local_files_only",
            type=int,
            default=1,
            help="Restrict Hugging Face loading to local files only (1=True, 0=False).",
        )
        parser.add_argument(
            "--timers1_batch_size",
            type=int,
            default=32,
            help="Number of per-target sequences generated in one Timer-S1 call.",
        )
        parser.add_argument(
            "--timers1_output_quantile",
            type=float,
            default=0.5,
            help="Quantile selected from Timer-S1 outputs as the benchmark point forecast.",
        )
        parser.add_argument(
            "--timers1_device",
            type=str,
            default="auto",
            choices=["auto", "cpu", "cuda"],
            help="Runtime placement preference for Timer-S1 loading/inference.",
        )
        parser.add_argument(
            "--timers1_use_revin",
            type=int,
            default=1,
            help="Enable Timer-S1 ReVIN/ReNorm preprocessing during generation (1=True, 0=False).",
        )

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        del x_dim, exo_con_list, exo_dis_dict
        return Model(timers1_local_files_only=bool(args.timers1_local_files_only), pred_len=pred_len, timers1_output_quantile=args.timers1_output_quantile, timers1_use_cache=bool(args.timers1_use_cache), timers1_model_path=args.timers1_model_path, timers1_use_revin=bool(args.timers1_use_revin), input_len=input_len, timers1_torch_dtype=args.timers1_torch_dtype, timers1_device=args.timers1_device, timers1_batch_size=args.timers1_batch_size, timers1_context_length=args.timers1_context_length)
