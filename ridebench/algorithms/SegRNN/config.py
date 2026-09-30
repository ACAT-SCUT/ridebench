from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class SegRNNConfig(ModelConfig):
    """SegRNN reproduction metadata.

    Paper: "Segment Recurrent Neural Network for Long-Term Time Series
    Forecasting" (IEEE IoT-J 2026).
    Paper URL: https://arxiv.org/abs/2308.11200
    Source code: https://github.com/lss-1138/SegRNN
    """

    @staticmethod
    def name() -> str:
        return "SegRNN"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument('--seg_len', metavar="LEN", type=int, default=48, help="length of the segment")
        parser.add_argument('--dropout', type=float, default=0.0, help="dropout rate")
        parser.add_argument('--rnn_type', metavar="TYPE", type=str, choices=['rnn', 'gru', 'lstm'], default='gru', help="type of RNN")
        parser.add_argument('--channel_id', metavar="ID", type=int, default=0, help="use channel id")
        parser.add_argument('--d_model', type=int, default=128, help="dimension of the hidden state")
        parser.add_argument('--dec_way', metavar="WAY", type=str, choices=['rmf', 'pmf'], default='pmf', help="decode method")
        parser.add_argument('--revin', type=int, default=1, help="use RevIN layer")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        config = Namespace()
        config.pred_len = pred_len
        config.enc_in = x_dim
        config.revin = args.revin
        config.dec_way = args.dec_way
        config.rnn_type = args.rnn_type
        config.seg_len = args.seg_len
        config.dropout = args.dropout
        config.d_model = args.d_model
        config.seq_len = input_len
        config.channel_id = args.channel_id
        return Model(config)
