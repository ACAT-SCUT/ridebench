from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class SparseTSFConfig(ModelConfig):
    """SparseTSF reproduction metadata.

    Conference paper: "SparseTSF: Modeling Long-term Time Series Forecasting
    with 1k Parameters" (ICML 2024 Oral).
    Journal extension: "SparseTSF: Lightweight and Robust Time Series
    Forecasting via Sparse Modeling" (TPAMI 2026).
    Paper URLs: https://arxiv.org/pdf/2405.00946 , https://ieeexplore.ieee.org/abstract/document/11141354
    Source code: https://github.com/lss-1138/SparseTSF
    """

    @staticmethod
    def name() -> str:
        return "SparseTSF"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument('--d_model', type=int, default=128, help="dimension of the hidden state")
        parser.add_argument('--period_len', metavar="LEN", type=int, default=48, help="length of the periodic sequence")
        parser.add_argument('--model_type', metavar="TYPE", type=str, choices=['linear', 'mlp'], default='mlp', help="type of NN model")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        config = Namespace()
        config.seq_len = input_len
        config.model_type = args.model_type
        config.pred_len = pred_len
        config.period_len = args.period_len
        config.d_model = args.d_model
        config.enc_in = x_dim
        return Model(config)
