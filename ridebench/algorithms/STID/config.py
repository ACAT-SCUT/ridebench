from argparse import Namespace

from ..config import ModelConfig
from .model import Model


class STIDConfig(ModelConfig):
    """STID reproduction metadata.

    Paper: "Spatial-Temporal Identity: A Simple yet Effective Baseline for
    Multivariate Time Series Forecasting" (CIKM 2022).
    Paper URL: https://arxiv.org/abs/2208.05233
    Source code: https://github.com/zezhishao/STID
    """

    @staticmethod
    def name() -> str:
        return "STID"

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument("--if_node", type=int, default=1, help="use endogenous-channel identity embeddings")
        parser.add_argument("--if_area", type=int, default=1, help="use benchmark area_id embeddings")
        parser.add_argument("--temp_dim_diw", type=int, default=8, help="dimension of intra-week embedding")
        parser.add_argument("--embed_dim", type=int, default=128, help="time-series embedding dimension")
        parser.add_argument("--if_D_i_W", type=int, default=1, help="use day_of_week embeddings")
        parser.add_argument("--node_dim", type=int, default=32, help="dimension of endogenous-channel identity embedding")
        parser.add_argument("--if_T_i_D", type=int, default=1, help="use half_hour_of_day embeddings")
        parser.add_argument("--area_dim", type=int, default=32, help="dimension of benchmark area identity embedding")
        parser.add_argument("--num_layer", type=int, default=1, help="number of residual MLP blocks")
        parser.add_argument("--temp_dim_tid", type=int, default=8, help="dimension of intra-day embedding")

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        del exo_con_list
        exo_dis_names = list(exo_dis_dict.keys())
        cfg = Namespace()
        cfg.node_dim = args.node_dim
        cfg.input_dim = 1
        cfg.area_dim = args.area_dim
        cfg.area_size = exo_dis_dict.get("area_id", 0)
        cfg.if_T_i_D = bool(args.if_T_i_D) and "half_hour_of_day" in exo_dis_dict
        cfg.time_of_day_size = exo_dis_dict.get("half_hour_of_day", 0)
        cfg.embed_dim = args.embed_dim
        cfg.tid_idx = exo_dis_names.index("half_hour_of_day") if "half_hour_of_day" in exo_dis_dict else None
        cfg.area_idx = exo_dis_names.index("area_id") if "area_id" in exo_dis_dict else None
        cfg.day_of_week_size = exo_dis_dict.get("day_of_week", 0)
        cfg.temp_dim_diw = args.temp_dim_diw
        cfg.output_len = pred_len
        cfg.num_layer = args.num_layer
        cfg.if_area = bool(args.if_area) and "area_id" in exo_dis_dict
        cfg.temp_dim_tid = args.temp_dim_tid
        cfg.diw_idx = exo_dis_names.index("day_of_week") if "day_of_week" in exo_dis_dict else None
        cfg.if_D_i_W = bool(args.if_D_i_W) and "day_of_week" in exo_dis_dict
        cfg.num_nodes = x_dim
        cfg.if_node = bool(args.if_node)
        cfg.input_len = input_len
        return Model(cfg)
