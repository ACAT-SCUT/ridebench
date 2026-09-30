from __future__ import annotations

from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


def add_covariate_arguments(parser: ArgumentParser) -> None:
    parser.add_argument(
        "--use_exog",
        type=int,
        choices=[0, 1],
        default=1,
        help="whether to use exogenous variables at all",
    )
    parser.add_argument(
        "--exog_source",
        type=str,
        choices=["none", "continuous", "discrete", "both"],
        default="both",
        help="which exogenous groups are enabled",
    )
    parser.add_argument(
        "--discrete_exog_mode",
        type=str,
        choices=["none", "ordinal", "onehot", "catemb"],
        default="onehot",
        help="discrete exogenous encoding: none, ordinal raw IDs, one-hot, or learnable categorical covariate embeddings",
    )
    parser.add_argument(
        "--catemb_dim",
        type=int,
        default=8,
        help="dimension of each learnable categorical covariate embedding",
    )
    parser.add_argument(
        "--future_exog_mode",
        type=str,
        choices=["auto", "disable", "enable"],
        default="auto",
        help="whether future-known exogenous variables are exposed to the model",
    )


def resolve_covariate_spec(
    x_dim: int,
    exo_con_list: list[str],
    exo_dis_dict: dict[str, int],
    args: Namespace,
    *,
    default_use_future_exog: bool,
    require_exog_channel: bool,
) -> SimpleNamespace:
    use_exog = bool(args.use_exog) and args.exog_source != "none"
    use_continuous = use_exog and args.exog_source in {"continuous", "both"}
    use_discrete = use_exog and args.exog_source in {"discrete", "both"} and args.discrete_exog_mode != "none"

    if args.future_exog_mode == "auto":
        use_future_exog = bool(default_use_future_exog)
    else:
        use_future_exog = args.future_exog_mode == "enable"

    continuous_dim = len(exo_con_list) if use_continuous else 0
    discrete_dim = _discrete_feature_dim(
        args.discrete_exog_mode,
        list(exo_dis_dict.values()),
        args.catemb_dim,
        use_discrete,
    )
    selected_dim = continuous_dim + discrete_dim
    model_exog_dim = selected_dim
    if require_exog_channel and model_exog_dim == 0:
        # Several reproduced covariate models structurally expect at least one
        # float exogenous channel. In benchmark PureEndo mode we therefore inject
        # a zero-valued dummy channel instead of rewriting the upstream blocks.
        model_exog_dim = 1

    return SimpleNamespace(
        x_dim=x_dim,
        use_exog=use_exog,
        use_continuous=use_continuous,
        use_discrete=use_discrete,
        discrete_exog_mode=args.discrete_exog_mode,
        catemb_dim=args.catemb_dim,
        use_future_exog=use_future_exog,
        continuous_dim=continuous_dim,
        discrete_dim=discrete_dim,
        selected_dim=selected_dim,
        model_exog_dim=model_exog_dim,
        series_dim=x_dim,
        enc_in=x_dim + model_exog_dim,
        discrete_cardinalities=list(exo_dis_dict.values()),
    )


def build_covariate_adapter(spec: SimpleNamespace) -> nn.Module:
    return CovariateAdapter(spec)


class CovariateAdapter(nn.Module):
    def __init__(self, spec: SimpleNamespace):
        super().__init__()
        self.spec = spec
        self.discrete_encoder = _build_discrete_encoder(spec)

    def forward(
        self,
        x_exo_con: torch.Tensor | None,
        x_exo_dis: torch.Tensor | None,
        y_exo_con: torch.Tensor | None,
        y_exo_dis: torch.Tensor | None,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        history = self._build_one_side(x_exo_con, x_exo_dis, dtype=dtype)
        future = self._build_one_side(y_exo_con, y_exo_dis, dtype=dtype)

        if self.spec.model_exog_dim == 0:
            return None, None

        if history is None:
            reference = _first_not_none(x_exo_con, x_exo_dis, y_exo_con, y_exo_dis)
            if reference is None:
                raise ValueError("Unable to infer exogenous tensor shape.")
            history_len = _infer_history_len(x_exo_con, x_exo_dis, y_exo_con, y_exo_dis)
            history = reference.new_zeros(
                (reference.shape[0], history_len, self.spec.model_exog_dim),
                dtype=dtype,
            )

        if future is None or not self.spec.use_future_exog:
            pred_len = 0
            if y_exo_con is not None:
                pred_len = y_exo_con.shape[1]
            elif y_exo_dis is not None:
                pred_len = y_exo_dis.shape[1]
            future = history.new_zeros(
                (history.shape[0], pred_len, self.spec.model_exog_dim),
                dtype=dtype,
            )

        return history, future

    def _build_one_side(
        self,
        exo_con: torch.Tensor | None,
        exo_dis: torch.Tensor | None,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        parts: list[torch.Tensor] = []
        if self.spec.use_continuous and exo_con is not None:
            parts.append(exo_con.to(dtype=dtype))
        if self.spec.use_discrete and exo_dis is not None and self.discrete_encoder is not None:
            parts.append(self.discrete_encoder(exo_dis, dtype=dtype))
        if not parts:
            return None
        return torch.cat(parts, dim=-1)


class OneHotDiscreteEncoder(nn.Module):
    def __init__(self, cardinalities: list[int]):
        super().__init__()
        self.cardinalities = cardinalities

    def forward(self, exo_dis: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        encoded = []
        for idx, num_classes in enumerate(self.cardinalities):
            feature = F.one_hot(exo_dis[..., idx], num_classes=num_classes).to(dtype=dtype)
            encoded.append(feature)
        return torch.cat(encoded, dim=-1)


class OrdinalDiscreteEncoder(nn.Module):
    def forward(self, exo_dis: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        # Benchmark ordinal mode intentionally keeps one scalar channel per
        # discrete feature and exposes the raw category ids as float inputs.
        return exo_dis.to(dtype=dtype)


class CatEmbDiscreteEncoder(nn.Module):
    def __init__(self, cardinalities: list[int], embed_dim: int):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, embed_dim) for cardinality in cardinalities]
        )

    def forward(self, exo_dis: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
        embedded = [embedding(exo_dis[..., idx]) for idx, embedding in enumerate(self.embeddings)]
        return torch.cat(embedded, dim=-1).to(dtype=dtype)


def _build_discrete_encoder(spec: SimpleNamespace) -> nn.Module | None:
    if not spec.use_discrete:
        return None
    if spec.discrete_exog_mode == "ordinal":
        return OrdinalDiscreteEncoder()
    if spec.discrete_exog_mode == "onehot":
        return OneHotDiscreteEncoder(spec.discrete_cardinalities)
    if spec.discrete_exog_mode == "catemb":
        # Benchmark adaptation: upstream models consume dense float exogenous
        # tensors only. CatEmb adds a learnable categorical covariate encoder
        # here while leaving the downstream reproduced architecture unchanged.
        return CatEmbDiscreteEncoder(
            spec.discrete_cardinalities,
            spec.catemb_dim,
        )
    raise ValueError(f"Unsupported discrete_exog_mode: {spec.discrete_exog_mode}")


def _discrete_feature_dim(
    mode: str,
    cardinalities: list[int],
    embed_dim: int,
    enabled: bool,
) -> int:
    if not enabled:
        return 0
    if mode == "ordinal":
        return len(cardinalities)
    if mode == "onehot":
        return sum(cardinalities)
    if mode == "catemb":
        return len(cardinalities) * embed_dim
    if mode == "none":
        return 0
    raise ValueError(f"Unsupported discrete_exog_mode: {mode}")


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _infer_history_len(
    x_exo_con: torch.Tensor | None,
    x_exo_dis: torch.Tensor | None,
    y_exo_con: torch.Tensor | None,
    y_exo_dis: torch.Tensor | None,
) -> int:
    if x_exo_con is not None:
        return x_exo_con.shape[1]
    if x_exo_dis is not None:
        return x_exo_dis.shape[1]
    if y_exo_con is not None:
        return y_exo_con.shape[1]
    if y_exo_dis is not None:
        return y_exo_dis.shape[1]
    raise ValueError("Unable to infer exogenous sequence length.")
