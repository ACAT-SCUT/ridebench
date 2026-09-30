from ..config import ModelConfig
from .model import Model


class SeasonalNaiveConfig(ModelConfig):
    """Seasonal naive baseline metadata.

    Baseline: copy the value from the previous seasonal position (`sp`) into the
    forecast horizon.
    Benchmark note: this wrapper is intentionally simple, uses no exogenous
    features, and the current simple-baseline scripts expose a single
    weekly-period variant.
    """

    @staticmethod
    def name() -> str:
        return "SeasonalNaive"

    @staticmethod
    def need_train() -> bool:
        return False

    @staticmethod
    def add_arguments(parser) -> None:
        parser.add_argument('--sp', type=int, default=48 * 7, help='Seasonal copy period in time steps; default 336 means one week for half-hour data.')

    @staticmethod
    def model(input_len, pred_len, x_dim, exo_con_list, exo_dis_dict, args):
        return Model(sp=args.sp, pred_len=pred_len)
