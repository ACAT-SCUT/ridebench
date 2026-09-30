from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class ModernTCNConfig(ModelConfig):
    """ModernTCN reproduction metadata.

    Paper: "ModernTCN: A Modern Pure Convolution Structure for General Time
    Series Analysis" (ICLR 2024 Spotlight).
    Paper URL: https://openreview.net/forum?id=vpJMJerXHU
    Source code: https://github.com/luodhhh/ModernTCN
    """

    @staticmethod
    def name() -> str:
        return "ModernTCN"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument('--patch_size', type=int, default=48, help='input patch length for the first convolutional stem')
        parser.add_argument('--dims', nargs='+', type=int, default=[256, 256, 256, 256], help='hidden dims per stage')
        parser.add_argument('--num_blocks', nargs='+', type=int, default=[1, 1, 1, 1], help='blocks per stage')
        parser.add_argument('--patch_stride', type=int, default=48, help='stride used by the first convolutional stem')
        parser.add_argument('--kernel_size', type=int, default=25, help='moving-average kernel size used by the decomposition branch')

        parser.add_argument('--individual', type=int, default=0, help='use one forecast head per channel instead of a shared head (1=True, 0=False)')
        parser.add_argument('--dw_dims', nargs='+', type=int, default=[256, 256, 256, 256], help='depthwise dims per stage')
        parser.add_argument('--revin', type=int, default=1, help='enable RevIN normalization (1=True, 0=False)')
        parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
        parser.add_argument('--small_size', nargs='+', type=int, default=[5, 5, 5, 5], help='small kernel per stage')

        parser.add_argument('--affine', type=int, default=0, help='enable affine parameters inside RevIN (1=True, 0=False)')
        parser.add_argument('--downsample_ratio', type=int, default=2, help='per-stage temporal downsampling ratio')
        parser.add_argument('--subtract_last', type=int, default=0, help='subtract the last observed value inside RevIN (1=True, 0=False)')
        parser.add_argument('--decomposition', type=int, default=0, help='enable the optional trend/seasonal decomposition branch (1=True, 0=False)')

        parser.add_argument('--ffn_ratio', type=int, default=2, help='hidden expansion ratio inside each TCN feed-forward block')
        parser.add_argument('--stem_ratio', type=int, default=6, help='channel expansion ratio used by the ModernTCN stem block')
        parser.add_argument('--freq', type=str, default='h', help='time feature frequency flag kept for upstream-compatible embeddings')
        parser.add_argument('--use_multi_scale', type=int, default=0, help='enable ModernTCN multi-scale feature fusion (1=True, 0=False)')
        parser.add_argument('--small_kernel_merged', action='store_true', help='indicate that re-parameterized small kernels have already been merged')
        parser.add_argument('--dropout', type=float, default=0.05, help='backbone dropout')
        parser.add_argument('--large_size', nargs='+', type=int, default=[31, 29, 27, 13], help='large kernel per stage')

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        config = Namespace()
        config.freq = args.freq
        config.num_blocks = args.num_blocks
        config.dw_dims = args.dw_dims
        config.downsample_ratio = args.downsample_ratio
        config.small_kernel_merged = bool(args.small_kernel_merged)
        config.head_dropout = args.head_dropout
        config.decomposition = bool(args.decomposition)
        config.small_size = args.small_size
        config.affine = bool(args.affine)
        config.use_multi_scale = bool(args.use_multi_scale)
        config.stem_ratio = args.stem_ratio
        config.dims = args.dims
        config.ffn_ratio = args.ffn_ratio
        config.large_size = args.large_size
        config.individual = bool(args.individual)
        config.subtract_last = bool(args.subtract_last)
        config.dropout = args.dropout
        config.seq_len = input_len
        config.kernel_size = args.kernel_size
        config.revin = bool(args.revin)
        config.patch_stride = args.patch_stride
        config.pred_len = pred_len
        config.enc_in = x_dim
        config.patch_size = args.patch_size
        return Model(config)
