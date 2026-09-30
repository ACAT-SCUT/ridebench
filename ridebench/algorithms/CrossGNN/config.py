from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class CrossGNNConfig(ModelConfig):
    """CrossGNN reproduction metadata.

    Paper: "CrossGNN: Confronting Noisy Multivariate Time Series Via Cross
    Interaction Refinement" (NeurIPS 2023).
    Paper URL: https://proceedings.neurips.cc/paper_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html
    Source code: https://github.com/hqh0728/CrossGNN
    """

    @staticmethod
    def name() -> str:
        return "CrossGNN"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument('--use_tgcn', type=int, default=1, help='use cross-scale gnn')
        parser.add_argument('--tk', type=int, default=10, help='topk for cross-scale neighbors')
        parser.add_argument('--anti_ood', type=int, default=1, help='anti distribution-shift trick')
        parser.add_argument('--e_layers', type=int, default=2, help='num of CrossGNN encoder layers')
        parser.add_argument('--gpu', type=int, default=0, help='gpu id used by CrossGNN internals')
        parser.add_argument('--use_ngcn', type=int, default=1, help='use cross-variable gnn')
        parser.add_argument('--hidden', type=int, default=8, help='channel hidden dim')
        parser.add_argument('--scale_number', type=int, default=4, help='scale number')
        parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
        parser.add_argument('--tvechidden', type=int, default=1, help='time vector hidden dim')
        parser.add_argument('--nvechidden', type=int, default=1, help='node vector hidden dim')
        parser.add_argument('--individual', type=int, default=0, help='individual linear head')

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        config = Namespace()
        config.use_tgcn = bool(args.use_tgcn)
        config.hidden = args.hidden
        config.enc_in = x_dim
        config.device = args.device
        config.use_ngcn = bool(args.use_ngcn)
        config.dropout = args.dropout
        config.individual = bool(args.individual)
        config.tk = args.tk
        config.anti_ood = bool(args.anti_ood)
        config.e_layers = args.e_layers
        config.nvechidden = args.nvechidden
        config.seq_len = input_len
        config.tvechidden = args.tvechidden
        config.pred_len = pred_len
        config.scale_number = args.scale_number
        return Model(config)
