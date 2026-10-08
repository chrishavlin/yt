import numpy as np

from yt.loaders import load_uniform_grid
from yt.testing import requires_module_pytest
from yt.utilities.on_demand_imports import _h5py as h5py
import pytest
import yt
from yt.testing import fake_amr_ds

@requires_module_pytest("h5py")
def test_save_as_data_unit_system(tmp_path):
    # This test checks that the file saved with calling save_as_dataset
    # for a ds with a "code" unit system contains the proper "unit_system_name".
    # It checks the hdf5 file directly rather than using yt.load(), because
    # https://github.com/yt-project/yt/issues/4315 only manifested restarting
    # the python kernel (because the unit registry is state dependent).

    fi = tmp_path / "output_data.h5"
    shp = (4, 4, 4)
    data = {"density": np.random.random(shp)}
    ds = load_uniform_grid(data, shp, unit_system="code")
    assert "code" in ds._unit_system_name

    sp = ds.sphere(ds.domain_center, ds.domain_width[0] / 2.0)
    sp.save_as_dataset(fi)

    with h5py.File(fi, mode="r") as f:
        assert f.attrs["unit_system_name"] == "code"


@requires_module_pytest("h5py")
@pytest.mark.parametrize(('selector_type', 'selector_args'), [
    ('sphere', ((0.5, 0.5, 0.5), .4,)),
    ('box', ((0.2,.2,.2),(0.8,0.8,0.8)))
])
def test_chunked_read_of_saved_subselection(tmp_path, selector_type, selector_args):


    # load data, build the selector
    ds = fake_amr_ds()
    selector = getattr(ds, selector_type)
    sp = selector(*selector_args)
    expected_shp = sp['stream', 'Density'].shape
    sp_sz = len(sp['stream', 'Density'])
    all_data_sz = len(ds.all_data()['stream','Density'])
    assert sp_sz < all_data_sz  # to make sure we've actually subselected.

    # save it
    fi = tmp_path / "output_data.h5"
    sp.save_as_dataset(fi, fields=ds.field_list)

    # reload it
    sp_ds = yt.load(fi)
    assert expected_shp == sp_ds.data[("grid", "Density")].shape
    assert len(sp_ds.data[("grid", "Density")]) < all_data_sz
    assert expected_shp == sp_ds.all_data()[("grid", "Density")].shape
