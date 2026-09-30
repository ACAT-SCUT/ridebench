from ..config import ModelConfig
from .model import Model


class SeasonalMeanConfig(ModelConfig):
    """Seasonal mean baseline metadata.

    Baseline: forecast each future seasonal position with the mean of all
    aligned seasonal positions observed in the input window.
    Benchmark note: this wrapper is endogenous-only and currently uses one
    weekly seasonal period for half-hour benchmark data.
    """

    @staticmethod
    def name() -> str:
        return "SeasonalMean"

    @staticmethod
    def need_train() -> bool:
        return False

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument(
            "--sp",
            type=int,
            default=48 * 7,
            help="Seasonal averaging period in time steps; default 336 means one week for half-hour data.",
        )

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        del input_len, x_dim, exo_con_list, exo_dis_dict
        return Model(sp=args.sp, pred_len=pred_len)
