from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class TimeMixerConfig(ModelConfig):
    """TimeMixer reproduction metadata.

    Paper: "TimeMixer: Decomposable Multiscale Mixing for Time Series
    Forecasting" (ICLR 2024).
    Paper URL: https://openreview.net/pdf?id=7oLshfEIC2
    Source code: https://github.com/kwuking/TimeMixer
    """

    @staticmethod
    def name() -> str:
        return "TimeMixer"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument('--e_layers', type=int, default=2, help='number of TimeMixer encoder layers')
        parser.add_argument('--enc_in', type=int, default=7, help='placeholder encoder input size; benchmark resolves the real endogenous channel count from endo_vars')
        parser.add_argument('--channel_independence', type=int, default=1,
                    help='0 enables channel mixing, 1 keeps the benchmark default channel-independent path')
        parser.add_argument('--moving_avg', type=int, default=25, help='moving-average kernel size used by the default decomposition path')
        parser.add_argument('--down_sampling_window', type=int, default=2, help='window size used by each multiscale downsampling step')
        parser.add_argument('--d_ff', type=int, default=256, help='feed-forward hidden size inside mixer blocks')
        parser.add_argument('--down_sampling_layers', type=int, default=1, help='number of multiscale downsampling layers')
        parser.add_argument('--top_k', type=int, default=5, help='top-k periods selected by the TimesBlock frequency module')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout used by embeddings and mixer blocks')
        parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='downsampling operator: avg, max, or conv')
        parser.add_argument('--label_len', type=int, default=48, help='decoder warmup length used by the reproduced TimeMixer interface')
        parser.add_argument('--d_model', type=int, default=128, help='model hidden size')
        parser.add_argument('--use_norm', type=int, default=1, help='enable normalization/denormalization inside TimeMixer (1=True, 0=False)')
        parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                    help='enable future temporal feature inputs in the upstream interface (1=True, 0=False)')
        parser.add_argument('--freq', type=str, default='h',
                    help='time feature frequency flag; supports standard aliases such as s/t/h/d/w/m and granular values like 15min or 3h')
        parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='series decomposition method; supported values are moving_avg and dft_decomp')
        parser.add_argument('--embed', type=str, default='timeF',
                    help='time feature embedding type: timeF, fixed, or learned')
        parser.add_argument('--c_out', type=int, default=7, help='placeholder output size; benchmark resolves the real target channel count from endo_vars')
        parser.add_argument('--task_name', type=str, default='long_term_forecast',
                    help='forecasting task type')

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        config = Namespace()
        config.seq_len = input_len
        config.down_sampling_layers = args.down_sampling_layers
        config.decomp_method = args.decomp_method
        config.use_norm = args.use_norm
        config.e_layers = args.e_layers
        config.dropout = args.dropout
        config.d_model = args.d_model
        config.moving_avg = args.moving_avg
        config.use_future_temporal_feature = args.use_future_temporal_feature
        config.d_ff = args.d_ff
        config.embed = args.embed
        config.channel_independence = args.channel_independence
        config.freq = args.freq
        config.down_sampling_method = args.down_sampling_method
        config.pred_len = pred_len
        config.label_len = args.label_len
        config.top_k = args.top_k
        config.c_out = x_dim
        config.enc_in = x_dim
        config.down_sampling_window = args.down_sampling_window
        config.task_name = args.task_name
        
        return Model(config)
