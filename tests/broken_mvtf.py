import numpy as np
import unyt

import yt
from yt.visualization.volume_rendering.transfer_functions import (
    MultiVariateTransferFunction,
    TransferFunction,
)

ds = yt.load("Enzo_64/DD0043/data0043")


dens_bounds = (-32, -27)


def _renormalized_dens(field, data):
    dens = np.log10(data["gas", "density"].copy().d)
    dens = ((dens - dens_bounds[0]) / (dens_bounds[1] - dens_bounds[0])) ** 2
    dens[dens < 0] = 0
    dens[dens > 1] = 0
    return unyt.unyt_array(dens, "")


ds.add_field(
    ("gas", "renormalized_density"),
    _renormalized_dens,
    sampling_type="local",
    units="",
    take_log=False,
    force_override=True,
)

T_bounds = (3, 8)


def _renormalized_T(field, data):
    vals = np.log10(data["gas", "temperature"].copy().d)
    vals = ((vals - T_bounds[0]) / (T_bounds[1] - T_bounds[0])) ** 2
    vals[vals < 0] = 0
    vals[vals > 1] = 0
    return unyt.unyt_array(vals, "")


ds.add_field(
    ("gas", "renormalized_temperature"),
    _renormalized_T,
    sampling_type="local",
    units="",
    take_log=False,
    force_override=True,
)


dens = ds.all_data()["gas", "renormalized_density"]
print(dens.min(), dens.max())


slc = yt.SlicePlot(ds, "x", ("gas", "renormalized_density"))
slc.save()

mv = MultiVariateTransferFunction()


green = TransferFunction((0, 1))
green.y = 0.5 * np.exp(-(((green.x - 0.5) / 0.1) ** 2))

red = TransferFunction((0, 1))
red.y = 0.5 * np.exp(-((red.x / 0.1) ** 25))

blue = TransferFunction((0, 1))
blue.y = 0.5 * np.exp(-(((blue.x - 0.1) / 0.25) ** 2))


# green = TransferFunction(T_bounds)
# green.y = 0 * np.exp(-(((green.x - 3.5) / 0.5) ** 2))

# red = TransferFunction(T_bounds)
# red.y = 1 - (red.x - T_bounds[0]) / (T_bounds[1] - T_bounds[0])
# red.y = red.y - 0.5 * green.y
# red.y[red.y < 0] = 0


# blue = TransferFunction(T_bounds)
# blue.y = (blue.x - T_bounds[0]) / (T_bounds[1] - T_bounds[0])
# blue.y = blue.y - 0.5 * green.y
# blue.y[blue.y < 0] = 0


for ich, tf in enumerate([red, green, blue]):
    mv.add_field_table(tf, 0, weight_field_id=1)
    mv.link_channels(ich, ich)


flds = [("gas", "renormalized_temperature"), ("gas", "renormalized_density")]
# renormed_dens_bounds = (0, 1)
alpha = TransferFunction((0.0, 1))
alpha.y = np.zeros(alpha.x.shape)
for ich in range(3):
    mv.add_field_table(alpha, 0, weight_field_id=1)
    mv.link_channels(ich + 3, ich + 3)


# alpha = TransferFunction(T_bounds)
# alpha.y = np.ones(alpha.x.shape)
# for ich in range(3):
#     mv.add_field_table(alpha, 0, weight_field_id=1)
#     mv.link_channels(ich + 3, ich + 3)

sc = yt.create_scene(ds.all_data(), flds)

cam = sc.camera
cam.set_resolution((1024, 1024))
sc[0].log_field = False
sc[0].weight_field = flds[1]
sc[0].log_weight_field = False
_ = sc[0].set_transfer_function(mv)

sc.save("mvtf_renormed_dens_wght.png")
