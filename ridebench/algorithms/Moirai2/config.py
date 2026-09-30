import os

from ..config import ModelConfig
from .model import Model


class Moirai2Config(ModelConfig):
    """Moirai 2.0 reproduction metadata.

    Paper: "Moirai 2.0: When Less Is More for Time Series Forecasting" (2025
    release).
    Paper URL: https://arxiv.org/abs/2511.11698
    Source code: https://github.com/SalesforceAIResearch/uni2ts
    Benchmark note: this wrapper follows the published Moirai 2.0 zero-shot
    usage and keeps a single endogenous-only variant. Benchmark exogenous
    variables are intentionally not wired into Moirai 2.0.
    """

    @staticmethod
    def name() -> str:
        return "Moirai2"

    @staticmethod
    def need_train() -> bool:
        return False

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument(
            "--use_exog",
            type=int,
            default=0,
            help="Unused for Moirai 2.0. Kept at 0 so the wrapper matches the official endogenous-only zero-shot usage.",
        )
        parser.add_argument(
            "--moirai_context_length",
            type=int,
            default=0,
            help="Optional context length cap; 0 uses input_len.",
        )
        parser.add_argument(
            "--moirai_device",
            type=str,
            default="auto",
            choices=["auto", "cpu", "cuda"],
            help="Runtime device used by the local Moirai 2.0 wrapper.",
        )
        parser.add_argument(
            "--moirai_local_files_only",
            type=int,
            default=0,
            help="Restrict Hugging Face fallback loading to local files only (1=True, 0=False).",
        )
        parser.add_argument(
            "--moirai_model_path",
            type=str,
            default=os.environ.get("BENCH_MOIRAI2_MODEL", "./pretrained_models/moirai-2.0-R-small"),
            help="Moirai 2.0 model id or local checkpoint directory.",
        )

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        del x_dim, exo_con_list, exo_dis_dict
        return Model(moirai_context_length=args.moirai_context_length, moirai_model_path=args.moirai_model_path, moirai_device=args.moirai_device, input_len=input_len, pred_len=pred_len, batch_size=args.batch_size, moirai_local_files_only=bool(args.moirai_local_files_only))
