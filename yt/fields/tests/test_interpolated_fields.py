import numpy as np
import pytest

from yt.fields.interpolated_fields import add_interpolated_field
from yt.testing import fake_amr_ds


@pytest.fixture
def ds_axesfields_table():
    ds = fake_amr_ds()

    shp = ds.domain_dimensions
    ndim = len(shp)
    table = np.random.random(shp)

    axes_fields = [("index", ds.coordinates.axis_order[id]) for id in range(ndim)]

    return ds, axes_fields, table


@pytest.mark.parametrize("axes_data_type", ("ndarray", "extent"))
def test_add_interpolated_fields(ds_axesfields_table, axes_data_type):
    """basic test that it runs"""

    ds, axes_fields, table = ds_axesfields_table

    le = ds.domain_left_edge.to("code_length").d
    re = ds.domain_right_edge.to("code_length").d
    if axes_data_type == "ndarray":
        # provide arrays for the table dimensions
        shp = ds.domain_dimensions
        ndim = len(le)
        axes_data = [np.linspace(le[id], re[id], shp[id]) for id in range(ndim)]
    else:
        # provide an extent tuple for the table dimensions
        axes_data = np.column_stack([le, re]).ravel().tolist()

    axes_data = tuple(axes_data)

    field = ("gas", "interpd_rand")
    add_interpolated_field(
        field,
        table,
        axes_data,
        axes_fields,
        ds=ds,
        units="",
        force_override=True,
    )

    assert field in ds.derived_field_list

    _ = ds.all_data()[field]
