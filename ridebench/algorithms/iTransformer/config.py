from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class ITransformerConfig(ModelConfig):
    """iTransformer reproduction metadata.

    Paper: "iTransformer: Inverted Transformers Are Effective for Time Series
    Forecasting" (ICLR 2024 Spotlight).
    Paper URL: https://arxiv.org/abs/2310.06625
    Source code: https://github.com/thuml/iTransformer
    """

    @staticmethod
    def name() -> str:
        return "iTransformer"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
        
        # model define
        parser.add_argument('--d_ff', type=int, default=256, help='feed-forward hidden size inside transformer blocks')
        parser.add_argument('--freq', type=str, default='h',
                        help='time feature frequency flag; supports standard aliases such as s/t/h/d/w/m and granular values like 15min or 3h')
        parser.add_argument('--output_attention', action='store_true', help='return encoder attention maps in addition to forecasts')
        parser.add_argument('--embed', type=str, default='timeF',
                        help='time feature embedding type: timeF, fixed, or learned')
        parser.add_argument('--activation', type=str, default='gelu', help='encoder activation function')
        parser.add_argument('--dropout', type=float, default=0.1, help='dropout used by embeddings, attention, and forecast head')
        parser.add_argument('--use_norm', type=int, default=True, help='enable normalization and denormalization inside iTransformer (1=True, 0=False)')
        parser.add_argument('--d_model', type=int, default=128, help='transformer hidden size')
        parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers')
        parser.add_argument('--factor', type=int, default=1, help='attention factor used by the reproduced FullAttention path')
        parser.add_argument('--class_strategy', type=str, default='projection', help='token aggregation strategy: projection, average, or cls_token')
        
        # iTransformer
        parser.add_argument('--label_len', type=int, default=48, help='decoder warmup length kept for upstream CLI compatibility')


    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        config = Namespace()
        config.factor = args.factor
        config.dropout = args.dropout
        config.use_norm = args.use_norm
        config.e_layers = args.e_layers
        config.n_heads = args.n_heads
        config.d_ff = args.d_ff
        config.seq_len = input_len
        config.output_attention = args.output_attention
        config.class_strategy = args.class_strategy
        config.freq = args.freq
        config.d_model = args.d_model
        config.pred_len = pred_len
        config.activation = args.activation
        config.embed = args.embed

        return Model(config)
