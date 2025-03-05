import numpy as np
import pytest

from yt import load_uniform_grid
from yt.fields.interpolated_fields import add_interpolated_field
from yt.testing import fake_amr_ds


@pytest.fixture
def ds_axesfields_table():
    ds = fake_amr_ds()

    shp = ds.domain_dimensions
    ndim = len(shp)
    ng = np.random.default_rng()
    table = ng.random(shp)

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

    # above field should only be registered with the above dataset, check it
    ds2 = fake_amr_ds()
    assert field not in ds2.derived_field_list


@pytest.mark.parametrize("ndim", (3,))
def test_interpolated_field_values(ndim):
    # create a dataset
    ng = np.random.default_rng()
    shp = (16,) * ndim
    dens = ng.random(shp)
    bbox = np.array([[0.0, 1.0] for _ in range(3)])

    if ndim == 2:
        dens = dens[:, :, np.newaxis]
        bbox[2, :] = [0.5, 0.5]
        shp = shp + (1,)

    ds = load_uniform_grid({"density": dens}, shp, bbox=bbox)
    assert ds.dimensionality == ndim

    # create a data field of same dimensions

    table = ng.random((16,) * ndim)
    axes_fields = tuple(
        [("index", ds.coordinates.axis_order[idim]) for idim in range(ndim)]
    )

    dxyz = ds.domain_width / ds.dimensionality
    bbox[:, 0] = bbox[:, 0] + dxyz.d / 2.0
    bbox[:, 1] = bbox[:, 1] - dxyz.d / 2.0
    axes_data = tuple(bbox.ravel()[: 2 * ndim].tolist())

    assert len(axes_data) == 2 * ndim
    assert len(axes_fields) == ndim

    fname = ("stream", "interp_field")
    add_interpolated_field(fname, table, axes_data, axes_fields, ds=ds)

    assert fname in ds.derived_field_list

    if ndim == 2:
        reg = ds.r[::16j, ::16j]
    else:
        reg = ds.r[::16j, ::16j, ::16j]
    vals = reg[fname]

    assert vals.shape == shp
    assert np.allclose(vals - table, 0.0)
    assert np.allclose(vals, table)
