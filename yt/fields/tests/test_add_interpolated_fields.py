import numpy as np

from yt.fields.interpolated_fields import add_interpolated_field
from yt.testing import fake_amr_ds


def test_add_interpolated_fields():
    ds = fake_amr_ds()

    shp = ds.domain_dimensions

    table = np.random.random(shp)

    le = ds.domain_left_edge.to("code_length").d
    re = ds.domain_right_edge.to("code_length").d
    ndim = len(le)
    axes_data = [np.linspace(le[id], re[id], shp[id]) for id in range(ndim)]
    axes_data = tuple(axes_data)

    axes_fields = [("index", ds.coordinates.axis_order[id]) for id in range(ndim)]

    field = ("gas", "interpd_rand")
    add_interpolated_field(field, table, axes_data, axes_fields, ds=ds, units="")

    assert field in ds.derived_field_list

    _ = ds.all_data()[field]
