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
    ftype: FieldType | None = None,
    particle_type: Any | None = None,
    validators: list[FieldValidator] | None = None,
    truncate: bool = True,
    *,
    sampling_type: str = "local",
    units: str | bytes | Unit | None = None,
    **kwargs,
):
    """add an interpolated field based on a data array

    Parameters
    ----------
    name : FieldKey | FieldName
        the field name to use for the interpolated field
    table_data : np.ndarray
        the data array.
    axes_data : tuple[np.ndarray,] | tuple[float,]
        either a tuple of numpy arrays definining the axis positions for
        each dimension of table_data or an extent tuple defining the bounds.
        If providing arrays, provide an array for each dimension of table_data
        specifying the positions for each axis. If providing an extent tuple,
        specifying the min and max bounds of each dimension in sequence, e.g.:
        (xmin, xmax, ymin, ymax, zmin, zmax).
    axes_fields : list[FieldKey]
        a list of dataset fields corresponding to the axes of the data
        table. For example, if the table corresponds to a field
        with spatial dimensions, this would be [('index', 'x'), ('index', 'y')
        and ('index', 'z')] for a 3d dataset and data table.
    ftype : FieldType | None, optional
        Deprecated, please specify a full (fytpe, fname) with the name
        argument.
    particle_type : Any | None, optional
        Deprecated and will error: no longer a valid argument.
    validators : list[FieldValidator] | None, optional
        an optional list of field validators
    truncate : bool, optional
        Whether to truncate the interpolated table for evalulating points
        outside the table, by default True. If False, errors will be raised
        when trying to evaluate the field outside its defined bounds.
    sampling_type : str, optional
        "cell" or "particle" or "local", default is "local".
    units : str | bytes | Unit | None, optional
        the units of the field.
    **kwargs:
        any additional keyword arguments accepted by add_field

    Examples
    --------

    First, load a dataset and create a numpy array for the data table


    >>> import yt
    >>> from yt.fields.interpolated_fields import add_interpolated_field
    >>> import numpy as np
    >>> ds = yt.load_sample("IsolatedGalaxy")
    >>> print(ds.domain_left_edge, d)

    >>> shp = (16, 16, 16)
    >>> table = np.random.random(shp)

    Now, specify how the dimensions of the table map to fields of the
    dataset. Most commonly, this will be the spatial coordinates:

    >>> axes_fields = (('index', 'x'), ('index', 'y'), ('index', 'z'))

    To specify the spatial position of the interpolated table, provide
    arrays for each axis of the data table. In this case, the arrays
    are set to cover the extent of the dataset:

    >>> axes_data = [np.linspace(0., 1., shp[idim]) for idim in range(3)]

    >>> add_interpolated_field(("gas", "my_field"),
    ...                         table,
    ...                         axes_data,
    ...                         axes_fields,
    ...                         ds=ds,
    ...                        )

    to instead provide axes_data as an extent tuple, provide the left and
    right edge of each dimension in sequence (xmin, xmax, ymin, ymax, zmin, zmax)

    >>> axes_data = (0., 1., 0., 1., 0., 1.)
    >>> add_interpolated_field(("gas", "my_field"),
    ...                         table,
    ...                         axes_data,
    ...                         axes_fields,
    ...                         ds=ds,
    ...                         force_override=True,
    ...                        )


    """
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

    if "ds" in kwargs:
        ds = kwargs.pop("ds")
        add_field_func = ds.add_field
    else:
        add_field_func = add_field

    add_field_func(
        fieldname,
        function=_interpolated_field,
        sampling_type=sampling_type,
        units=units,
        validators=validators,
        **kwargs,
    )
