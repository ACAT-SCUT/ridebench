"""Translate dataset batches into the common five-input model contract."""

INPUT_FIELDS = ("x_endo", "x_exo_con", "x_exo_dis", "y_exo_con", "y_exo_dis")
DISCRETE_FIELDS = frozenset({"x_exo_dis", "y_exo_dis"})


def unpack_batch(batch, convert):
    return tuple(convert(name, batch[name]) for name in INPUT_FIELDS), convert("y", batch["y"])


def torch_batch(batch, device, dtype):
    def convert(name, value):
        options = {"device": device}
        if name not in DISCRETE_FIELDS:
            options["dtype"] = dtype
        return value.to(**options)
    return unpack_batch(batch, convert)


def numpy_batch(batch, dtype):
    def convert(name, value):
        array = value.numpy(force=True)
        return array if name in DISCRETE_FIELDS else array.astype(dtype)
    return unpack_batch(batch, convert)
