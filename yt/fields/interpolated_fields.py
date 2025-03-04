from typing import Any

import numpy as np

from yt._typing import FieldKey, FieldName, FieldType, Unit
from yt.fields.derived_field import FieldValidator
from yt.fields.local_fields import add_field
from yt.utilities.linear_interpolators import (
    BilinearFieldInterpolator,
    TrilinearFieldInterpolator,
    UnilinearFieldInterpolator,
)

_int_class = {
    1: UnilinearFieldInterpolator,
    2: BilinearFieldInterpolator,
    3: TrilinearFieldInterpolator,
}


def add_interpolated_field(
    name: FieldKey | FieldName,
    table_data: "np.ndarray",
    axes_data: tuple["np.ndarray",] | tuple[float,],
    axes_fields: list[FieldKey],
    ftype: FieldType = None,
    particle_type: Any | None = None,
    validators: list[FieldValidator] | None = None,
    truncate: bool = True,
    *,
    sampling_type: str = "local",
    ds=None,
    units: str | bytes | Unit | None = None,
    **kwargs,
):
    if isinstance(name, tuple) and ftype is not None:
        msg = "Do not specify ftype when providing a full field tuple"
        raise RuntimeError(msg)
    elif isinstance(name, str):
        ftype = ftype or "gas"  # preserve prior behavior
        fieldname = (ftype, name)
        msg = "The ftype argument is now deprecated, please specify a full field name tuple (name, fieldtype)"
        raise DeprecationWarning(msg)
    else:
        fieldname = name

    if particle_type is not None:
        # note: particle_type raises an error with add_fields, so it is not
        # feasible to use a deprecation warning.
        raise RuntimeError("particle_type is no longer a valid argument.")

    if len(table_data.shape) not in _int_class:
        raise RuntimeError(
            "Interpolated field can only be created from 1d, 2d, or 3d data."
        )

    ndim = table_data.ndim

    if isinstance(axes_data[0], float) and len(axes_data) != ndim * 2:
        # we're dealing with an extent tuple and the bounding box is not the right size
        msg = f"Data dimension mismatch: data is {ndim}, and {len(axes_data)} "
        msg += f"provided. Expected {ndim * 2} values for axes_data when specifying "
        msg += "axes_data as an extent."
        raise RuntimeError(msg)
    elif isinstance(axes_data[0], np.ndarray) and len(axes_data) != ndim:
        raise RuntimeError(
            f"Data dimension mismatch: data is {ndim}, "
            f"{len(axes_data)} axes data provided, "
            f"and {len(axes_fields)} axes fields provided."
        )

    if len(axes_fields) != ndim:
        raise RuntimeError(
            f"Data dimension mismatch: data is {ndim}, "
            f"{len(axes_data)} axes data provided, "
            f"and {len(axes_fields)} axes fields provided."
        )

    int_class = _int_class[len(table_data.shape)]
    my_interpolator = int_class(table_data, axes_data, axes_fields, truncate=truncate)

    def _interpolated_field(field, data):
        return my_interpolator(data)

    add_field(
        fieldname,
        function=_interpolated_field,
        sampling_type=sampling_type,
        units=units,
        validators=validators,
        ds=ds,
        **kwargs,
    )
