import os
import stat
import weakref
from collections import OrderedDict

import numpy as np

from yt.data_objects.index_subobjects.grid_patch import AMRGridPatch
from yt.data_objects.static_output import Dataset
from yt.geometry.grid_geometry_handler import GridIndex
from yt.utilities.file_handler import NetCDF4FileHandler, warn_netcdf
from yt.utilities.logger import ytLogger as mylog

from .fields import IRISFieldInfo


class IRISGrid(AMRGridPatch):
    _id_offset = 0

    def __init__(self, id, index, level):
        super(IRISGrid, self).__init__(id, filename=index.index_filename, index=index)
        self.Parent = None
        self.Children = []
        self.Level = level

    def __repr__(self):
        return "IRISGrid_%04i (%s)" % (self.id, self.ActiveDimensions)


class IRISHierarchy(GridIndex):
    grid = IRISGrid

    def __init__(self, ds, dataset_type="cm1"):
        self.dataset_type = dataset_type
        self.dataset = weakref.proxy(ds)
        # for now, the index file is the dataset!
        self.index_filename = self.dataset.parameter_filename
        self.directory = os.path.dirname(self.index_filename)
        # float type for the simulation edges and must be float64 now
        self.float_type = np.float64
        super(IRISHierarchy, self).__init__(ds, dataset_type)

    def _initialize_state_variables(self):
        super(IRISHierarchy, self)._initialize_state_variables()
        self.num_grids = 1

    def _detect_output_fields(self):
        # build list of on-disk fields for dataset_type 'cm1'
        self.field_list = []
        for vname in self.dataset.parameters["variable_names"]:
            self.field_list.append(("cm1", vname))

    def _count_grids(self):
        self.num_grids = 1

    def _parse_index(self):
        self.grid_left_edge[0][:] = self.ds.domain_left_edge[:]
        self.grid_right_edge[0][:] = self.ds.domain_right_edge[:]
        self.grid_dimensions[0][:] = self.ds.domain_dimensions[:]
        self.grid_particle_count[0][0] = 0
        self.grid_levels[0][0] = 1
        self.max_level = 1

    def _populate_grid_objects(self):
        self.grids = np.empty(self.num_grids, dtype="object")
        for i in range(self.num_grids):
            g = self.grid(i, self, self.grid_levels.flat[i], self.grid_dimensions[i])
            g._prepare_grid()
            g._setup_dx()
            self.grids[i] = g


class IRISmeta(object):
    def __init__(self, nc_ds):
        # copy over all non-callable attributes, excluding variables
        for attr in dir(nc_ds):
            if callable(getattr(nc_ds, attr)) is False and attr != "variables":
                setattr(self, attr, getattr(nc_ds, attr))


class IRISDataset(Dataset):
    _index_class = IRISHierarchy
    _field_info_class = IRISFieldInfo

    def __init__(
        self, filename, dataset_type="IRIS", storage_filename=None, units_override=None,
    ):
        self.fluid_types += ("IRIS",)
        self._handle = NetCDF4FileHandler(filename)
        super(IRISDataset, self).__init__(
            filename, dataset_type, units_override=units_override, unit_system="mks"
        )
        self.storage_filename = storage_filename
        # refinement factor between a grid and its subgrid
        # self.refine_by = 2

    def _set_code_unit_attributes(self):
        self.length_unit = self.quan(1.0, "km")
        self.mass_unit = self.quan(1.0, "kg")
        self.time_unit = self.quan(1.0, "s")
        self.time_unit = self.quan(1.0, "s")
        self.velocity_unit = self.quan(1.0, "km/s")
        pass

    def _parse_parameter_file(self):

        #   self.geometry  <= a lower case string
        #                     ("cartesian", "polar", "cylindrical"...)
        #                     (defaults to 'cartesian')
        self.unique_identifier = int(os.stat(self.parameter_filename)[stat.ST_CTIME])
        self.geometry = "cartesian"  # actually "internal_geographic"...
        self.parameters = {}  # code-specific items
        with self._handle.open_ds() as ds:
            dims = [ds.dimensions[i].size for i in ds.dimensions]
            x, y, z = [ds.variables[i][:] for i in ds.dimensions]
            self.domain_left_edge = np.array(
                [x.min(), y.min(), z.min()], dtype="float64"
            )
            self.domain_right_edge = np.array(
                [x.max(), y.max(), z.max()], dtype="float64"
            )

            # loop over the variables in the netCDF file, record meta data for each
            var_meta = {}
            metakeys = [
                "name",
                "dimensions",
                "shape",
                "ndim",
                "mask",
                "dtype",
                "long_name",
                "missing_value",
            ]
            for varname in ds.variables:
                var = ds.variables[varname]
                var_meta[varname] = {i: getattr(var, i, None) for i in metakeys}

            self.parameters["variables"] = var_meta
            self.current_time = 0.0

            # record the dimension metadata
            dim_info = OrderedDict()
            for dim, meta in ds.dimensions.items():
                dim_info[dim] = meta.size
            self.parameters["dimensions"] = dim_info
            self.parameters["meta"] = IRISmeta(ds)

        self.dimensionality = len(dims)
        self.domain_dimensions = np.array(dims, dtype="int64")
        self.periodicity = (False,) * self.dimensionality
        self.parameters["time"] = self.current_time

        # Set cosmological information to zero for non-cosmological.
        self.cosmological_simulation = 0.0
        self.current_redshift = 0.0
        self.omega_lambda = 0.0
        self.omega_matter = 0.0
        self.hubble_constant = 0.0
        pass

    @classmethod
    def _is_valid(self, *args, **kwargs):
        warn_netcdf(args[0])
        try:
            from netCDF4 import Dataset

            filename = args[0]
            with Dataset(filename, keepweakref=True) as f:
                if "IRIS" in f.repository_institution:
                    mylog.info("loading IRIS EMC netcdf file.")
                    return True
        except Exception:
            pass
