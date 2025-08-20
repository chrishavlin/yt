import numpy as np

import yt
from yt.visualization.volume_rendering.transfer_functions import (
    MultiVariateTransferFunction,
    TransferFunction,
)

ds = yt.load("Enzo_64/DD0043/data0043")


flds = [("gas", "temperature"), ("gas", "density")]
T_bounds = (1, 8)
dens_bounds = (-34, -27)

mv = MultiVariateTransferFunction()

red = TransferFunction(T_bounds)
red.y = 1 - (red.x - T_bounds[0]) / (T_bounds[1] - T_bounds[0])
red.y = red.y - 0.5 * np.exp(-(((red.x - 3.5) / 0.5) ** 2))
red.y[red.y < 0] = 0

green = TransferFunction(T_bounds)
green.y = np.exp(-(((green.x - 3.5) / 0.5) ** 2))

blue = TransferFunction(T_bounds)
blue.y = (blue.x - T_bounds[0]) / (T_bounds[1] - T_bounds[0])
blue.y = blue.y - 0.5 * np.exp(-(((blue.x - 3.5) / 0.5) ** 2))
blue.y[blue.y < 0] = 0

for ich, tf in enumerate([red, green, blue]):
    mv.add_field_table(tf, 0)
    mv.link_channels(ich, ich)


alpha = TransferFunction(dens_bounds)
alpha.y = np.zeros(alpha.x.shape)
for ich in range(3):
    mv.add_field_table(alpha, 1)
    mv.link_channels(ich + 3, ich + 3)

sc = yt.create_scene(ds.all_data(), flds)
sc[0].set_weight_field(flds[1])
_ = sc[0].set_transfer_function(mv)
im = sc.render()
