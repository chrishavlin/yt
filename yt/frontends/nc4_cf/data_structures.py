import os
import stat
import weakref
from collections import OrderedDict

import numpy as np

from yt.data_objects.index_subobjects.grid_patch import AMRGridPatch
from yt.data_objects.static_output import Dataset
from yt.geometry.grid_geometry_handler import GridIndex
from yt.utilities.file_handler import NetCDF4FileHandler, warn_netcdf

from .fields import NCCFFieldInfo

# define some common dimension names to check against
x_names = ['longitude', 'lon', 'long']
y_names = ['latitude', 'lat', 'latg']
geo_names = x_names + y_names
internal_names = ['depth', 'radius']

class NCCFGrid(AMRGridPatch):
    _id_offset = 0

    def __init__(self, id, index, level, dimensions):
        super(NCCFGrid, self).__init__(id, filename=index.index_filename, index=index)
        self.Parent = None
        self.Children = []
        self.Level = level
        self.ActiveDimensions = dimensions

    def __repr__(self):
        return f"NCCFGrid_{self.id:d} ({self.ActiveDimensions})"


class NCCFHierarchy(GridIndex):
    grid = NCCFGrid

    def __init__(self, ds, dataset_type="nc4_cf"):
        self.dataset_type = dataset_type
        self.dataset = weakref.proxy(ds)
        # for now, the index file is the dataset!
        self.index_filename = self.dataset.parameter_filename
        self.directory = os.path.dirname(self.index_filename)
        # float type for the simulation edges and must be float64 now
        self.float_type = np.float64
        super(NCCFHierarchy, self).__init__(ds, dataset_type)

    def _detect_output_fields(self):
        # build list of on-disk fields for dataset_type 'nccf'
        vnames = self.dataset.parameters["variable_names"]
        self.field_list = [(self.dataset_type, vname) for vname in vnames]

    def _count_grids(self):
        # This needs to set self.num_grids

        # different vars could have different dims, so we need different grids...
        self.num_grids = 1

    def _parse_index(self):
        dim = self.dataset.dimensionality
        self.grid_left_edge[0][:dim] = self.ds.domain_left_edge[:]
        self.grid_right_edge[0][:dim] = self.ds.domain_right_edge[:]
        self.grid_dimensions[0][:dim] = self.ds.domain_dimensions[:]
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


class NCCFDataset(Dataset):
    _index_class = NCCFHierarchy
    _field_info_class = NCCFFieldInfo
    _min_version = 1.8

    def __init__(
        self,
        filename,
        dataset_type="nc4_cf",
        storage_filename=None,
        units_override=None,
        unit_system="mks",
        time_vars=['time'],
    ):
        self.fluid_types += (dataset_type,)
        self._handle = NetCDF4FileHandler(filename)
        # refinement factor between a grid and its subgrid
        self.refine_by = 1
        self.time_vars = time_vars
        super(NCCFDataset, self).__init__(
            filename,
            dataset_type,
            units_override=units_override,
            unit_system=unit_system,
        )
        self.storage_filename = storage_filename
        self.filename = filename

    def _set_code_unit_attributes(self):
        # This is where quantities are created that represent the various
        # on-disk units.  These are the currently available quantities which
        # should be set, along with examples of how to set them to standard
        # values.

        # with self._handle.open_ds() as _handle:
        #     length_unit = _handle.variables["xh"].units
        self.length_unit = self.quan(1.0, "m")
        self.mass_unit = self.quan(1.0, "kg")
        self.time_unit = self.quan(1.0, "s")
        self.velocity_unit = self.quan(1.0, "m/s")
        self.time_unit = self.quan(1.0, "s")

    def _infer_geometry(self):
        # checks if this dataset is a geographic or internal_geographic dataset,
        # defaulting to cartesian
        is_internal = 0

        with self._handle.open_ds() as _handle:
            dims = list(_handle.dimensions.keys())
            is_geographic = sum([dim.lower() in geo_names for dim in dims])
            if is_geographic:
                is_internal = sum([dim.lower() in internal_names for dim in dims])

        if is_internal:
            ordering = ('depth', 'latitude', 'longitude')  # this should be inferred as well...
            geometry = ('internal_geographic', ordering)
        elif is_geographic:
            ordering = ('altitude', 'latitude', 'longitude')  # this should be inferred as well...
            geometry = ('geographic', ordering)
        else:
            geometry = 'cartesian'

        return geometry

    def _infer_periodicity(self, dims, min_vals, max_vals):
        """
        Infers the periodicity of the dataset from lat/lon ranges

        dims : tuple of dimension names, ('lat','lon','z')
        min_vals : list of min values for each dimension
        max_vals : list of max values for each dimension
        """

        periodicity = []
        for idim, dim in enumerate(dims):
            dist = max_vals[idim] - min_vals[idim]
            if dim in x_names and dist > 359.:
                periodicity.append(True)
            elif dim in y_names and dist > 179.:
                periodicity.append(True)
            else:
                periodicity.append(False)

        if self.dimensionality == 2:
            periodicity.append(False)

        self.periodicity = tuple(periodicity)

    def _set_dimension_hashes(self, _handle):
        # builds several hashes and lists to easily retrieve
        # dimensionality for a field. Not all of these are used right now...
        var2dim = {}  # e.g., var2dim['cp']=('latitude','longitude')
        dim_set = set()  # set of spatial dims, should match the number of grids
        # dict of lists with fieldnames by dimensionality:
        vars_by_dim = {i: [] for i in range(4)}
        time_index = {} # the time index for each field

        for field in _handle.variables:
            if field not in _handle.dimensions:
                dims = _handle.variables[field].dimensions
                full_dims = tuple(i for i in dims if i not in self.time_vars)
                time_var = [i for i in dims if i in self.time_vars]
                if len(time_var):
                    time_index[field] = dims.index(time_var[0])
                # full_dims, _, _, _ = self._normalize_dims(full_dims)
                vars_by_dim[len(full_dims)].append(field)
                if len(full_dims) > 0:
                    var2dim[field] = full_dims
                    dim_set.update((full_dims,))

        if len(dim_set) > 1:
            # map the different sets to different grids?
            print("add a warning about multiple grids...")

        self.parameters["dim_set"] = dim_set

        # set a primary dimension set
        dim_list = list(dim_set)
        primary_dim = dim_list[0]
        for dim in dim_list:
            if len(dim) > len(primary_dim):
                primary_dim = dim
        self.parameters["primary_dim"] = primary_dim
        self.parameters["variable_dimensions"] = var2dim
        self.parameters["vars_by_dim"] = vars_by_dim
        self.parameters["time_index"] = time_index
        if len(vars_by_dim[3]) == 0:
            self.dimensionality = 2
        else:
            self.dimensionality = 3

    def _set_domain_extents(self, _handle):
        # sets domain limits and dimensions
        dims = self.parameters["primary_dim"]
        xyz = [_handle.variables[i][:] for i in dims]
        min_vals = [xyz[i].min() for i in range(self.dimensionality)]
        max_vals = [xyz[i].max() for i in range(self.dimensionality)]

        # domain dims have to be 3d, even if 2d:
        if self.dimensionality == 2:
            min_vals.append(0.0)
            max_vals.append(1.0)

        self._infer_periodicity(dims, min_vals, max_vals)
        self.domain_left_edge = np.array(min_vals, dtype="float64")
        self.domain_right_edge = np.array(max_vals, dtype="float64")
        ndims = [_handle.dimensions[i].size for i in dims]
        self.domain_dimensions = np.array(ndims, dtype="int64")

    def _check_uniform_mesh(self, _handle):
        # returns True if uniform, False is non-uniform
        dims = list(self.parameters["dim_set"])[0]
        if hasattr(_handle, "uniform_mesh"):
            return _handle.uniform_mesh
        else:
            for dim in [_handle.variables[i][:] for i in dims]:
                d_dim_all = dim[1:] - dim[:-1]
                d_dim = np.full(d_dim_all.shape, dim[1] - dim[0])
                if not np.allclose(d_dim_all, d_dim):
                    msg = "netcdf data must be on a uniform grid at present."
                    raise NotImplementedError(msg)
        return True

    def _parse_parameter_file(self):
        # This needs to set up the following items.  Note that these are all
        # assumed to be in code units; domain_left_edge and domain_right_edge
        # will be converted to YTArray automatically at a later time.
        # This includes the cosmological parameters.
        #
        #   self.unique_identifier      <= unique identifier for the dataset
        #                                  being read (e.g., UUID or ST_CTIME)
        self.unique_identifier = int(os.stat(self.parameter_filename)[stat.ST_CTIME])
        self.parameters = {}  # code-specific items

        with self._handle.open_ds() as _handle:
            # _handle here is a netcdf Dataset object, we need to parse some metadata
            # for constructing our yt ds.

            self._set_dimension_hashes(_handle)
            self._set_domain_extents(_handle)
            self.parameters["variable_names"] = list(_handle.variables)
            self.parameters["is_uniform"] = self._check_uniform_mesh(_handle)
            self.current_time = _handle.variables["time"][:][0]

            # record the dimension metadata: __handle.dimensions contains netcdf
            # objects so we need to manually copy over attributes.
            dim_info = OrderedDict()
            for dim, meta in _handle.dimensions.items():
                dim_info[dim] = meta.size
            self.parameters["dimensions"] = dim_info

        self.geometry = self._infer_geometry()
        self._zero_cosmological_constants()

    def _zero_cosmological_constants(self):
        # Set cosmological information to zero for non-cosmological dataset.
        self.cosmological_simulation = 0
        self.current_redshift = 0.0
        self.omega_lambda = 0.0
        self.omega_matter = 0.0
        self.hubble_constant = 0.0

    @classmethod
    def _is_valid(cls, filename, *args, **kwargs):
        # This accepts a filename or a set of arguments and returns True or
        # False depending on if the file is of the type requested.

        warn_netcdf(filename)
        min_version = 1.0
        try:
            nc4_file = NetCDF4FileHandler(filename)
            with nc4_file.open_ds(keepweakref=True) as _handle:
                # cf compliant netcdf files should have
                CFcon = getattr(
                    _handle, "Conventions", None
                )  # should be, e.g., "CF-1.8"
                if CFcon and "CF-" in CFcon:
                    version = float(CFcon.split("-")[-1])
                    if version < min_version:
                        return False  # should print a warning too
                else:
                    return False
        except (OSError, ImportError):
            return False

        return True
