from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class DLinearConfig(ModelConfig):
    """DLinear reproduction metadata.

    Paper: "Are Transformers Effective for Time Series Forecasting?" (AAAI
    2023).
    Paper URL: https://arxiv.org/pdf/2205.13504.pdf
    Source code: https://github.com/cure-lab/LTSF-Linear
    """

    @staticmethod
    def name() -> str:
        return "DLinear"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument('--individual', type=int, choices=[0, 1], default=0, help="1=true,0=false")
        parser.add_argument('--kernel_size', type=int, default=25, help="odd decomposition kernel size")


    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        config = Namespace()
        config.pred_len = pred_len
        config.enc_in = x_dim
        config.seq_len = input_len
        config.kernel_size = args.kernel_size
        config.individual = bool(args.individual)
        return Model(config)
