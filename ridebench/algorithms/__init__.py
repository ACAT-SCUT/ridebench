from importlib import import_module

_CONFIG_SPECS = [
    ("CATS", "CATSConfig"),
    ("PMDformer", "PMDformerConfig"),
    ("TQNet", "TQNetConfig"),
    ("SparseTSF", "SparseTSFConfig"),
    ("TimeBase", "TimeBaseConfig"),
    ("SegRNN", "SegRNNConfig"),
    ("SOFTS", "SOFTSConfig"),
    ("STID", "STIDConfig"),
    ("TimerS1", "TimerS1Config"),
    ("iTransformer", "ITransformerConfig"),
    ("Leddam", "LeddamConfig"),
    ("TimeMixer", "TimeMixerConfig"),
    ("TiDE", "TiDEConfig"),
    ("PhaseFormer", "PhaseFormerConfig"),
    ("TimeFilter", "TimeFilterConfig"),
    ("ModernTCN", "ModernTCNConfig"),
    ("DLinear", "DLinearConfig"),
    ("PatchTST", "PatchTSTConfig"),
    ("PETformer", "PETformerConfig"),
    ("RMLP", "RMLPConfig"),
    ("Moirai2", "Moirai2Config"),
    ("SeasonalNaive", "SeasonalNaiveConfig"),
    ("SeasonalMean", "SeasonalMeanConfig"),
    ("Chronos2", "Chronos2Config"),
    ("TimesFM2p5", "TimesFM2p5Config"),
    ("TimeXer", "TimeXerConfig"),
    ("XLinear", "XLinearConfig"),
    ("CrossLinear", "CrossLinearConfig"),
    ("DAG", "DAGConfig"),
    ("DUET", "DUETConfig"),
    ("CrossGNN", "CrossGNNConfig"),
    ("Crossformer", "CrossformerConfig"),
]

configs = []
failed_configs = {}

for module_name, class_name in _CONFIG_SPECS:
    try:
        module = import_module(f".{module_name}.config", __name__)
        configs.append(getattr(module, class_name))
    except Exception as exc:  # Optional model deps can fail at import time.
        failed_configs[module_name] = str(exc)
