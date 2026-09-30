"""Declarative experiment catalog used by the local task scheduler."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Task:
    group: str
    label: str
    command: tuple[str, ...]


REGULAR_GROUPS = {
    "simple": ("benchmark/simple/SeasonalNaive.sh", "benchmark/simple/SeasonalMean.sh"),
    "independent": tuple(f"benchmark/independent/{name}.sh" for name in (
        "SparseTSF_MLP", "DLinear", "PatchTST", "SegRNN", "PETformer", "PhaseFormer",
        "RMLP", "STID", "TimeBase", "TimeMixer",
    )),
    "multivariate": tuple(f"benchmark/multivariate/{name}.sh" for name in (
        "Crossformer", "CrossGNN", "DUET", "Leddam", "ModernTCN", "PMDformer",
        "SOFTS", "TimeFilter", "TQNet", "iTransformer",
    )),
    "exogenous": tuple(f"benchmark/exogenous/{name}.sh" for name in (
        "CATS_PureEndo", "CATS_Ordinal", "CATS_OneHot", "CATS_CatEmb",
        "CrossLinear_PureEndo", "CrossLinear_Ordinal", "CrossLinear_OneHot", "CrossLinear_CatEmb",
        "DAG_PureEndo", "DAG_Ordinal", "DAG_OneHot", "DAG_CatEmb",
        "TiDE_PureEndo", "TiDE_Ordinal", "TiDE_OneHot", "TiDE_CatEmb",
        "TimeXer_PureEndo", "TimeXer_Ordinal", "TimeXer_OneHot", "TimeXer_CatEmb",
        "XLinear_PureEndo", "XLinear_Ordinal", "XLinear_OneHot", "XLinear_CatEmb",
    )),
    "foundation": tuple(f"benchmark/foundation/{name}.sh" for name in (
        "Chronos2_NoExo", "Chronos2_FullExo", "Moirai2", "TimesFM2p5", "TimerS1",
    )),
}

STUDY_GROUPS = {
    "loss": tuple(f"study_loss/{name}.sh" for name in (
        "PETformer", "PatchTST", "PMDformer", "TimeFilter", "CrossLinear_OneHot", "TiDE_OneHot",
    )),
    "area": tuple(f"study_area/{name}.sh" for name in (
        "PETformer", "PatchTST", "PMDformer", "TimeFilter", "CrossLinear_OneHot", "TiDE_OneHot",
        "Chronos2_FullExo", "TimerS1",
    )),
    "lookback": tuple(f"study_lookback/{name}.sh" for name in (
        "PETformer", "PatchTST", "PMDformer", "TimeFilter", "CrossLinear_OneHot", "TiDE_OneHot",
        "Chronos2_FullExo", "TimerS1",
    )),
}


def _shell_task(group, script):
    path = Path(script)
    return Task(group, f"{group}/{path.stem}", ("bash", f"scripts/{script}"))


def regular_tasks(groups=None):
    selected = tuple(REGULAR_GROUPS) if groups is None else tuple(groups)
    return [_shell_task("regular", script) for group in selected for script in REGULAR_GROUPS[group]]


def ultra_long_tasks(groups=None):
    selected = tuple(REGULAR_GROUPS) if groups is None else tuple(groups)
    return [Task("ultra_long", f"ultra_long/{Path(script).stem}",
                 ("bash", "scripts/benchmark/ultra_long_forecast/run_single.sh", Path(script).stem))
            for group in selected for script in REGULAR_GROUPS[group]]


def study_tasks(study):
    return [_shell_task(study, script) for script in STUDY_GROUPS[study]]


def all_tasks():
    return regular_tasks() + ultra_long_tasks() + [task for study in STUDY_GROUPS for task in study_tasks(study)]


def tasks_for_suite(suite, groups=None):
    if suite in ("regular", "ultra_long"):
        return regular_tasks(groups) if suite == "regular" else ultra_long_tasks(groups)
    if suite in STUDY_GROUPS:
        return study_tasks(suite)
    if suite == "all":
        if groups:
            raise ValueError("--groups is only supported for regular and ultra_long suites.")
        return all_tasks()
    raise ValueError(f"Unknown suite: {suite}")
