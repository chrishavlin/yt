"""
CF Radial data structures



"""

import os
import weakref

import numpy as np

from yt.data_objects.index_subobjects.grid_patch import AMRGridPatch
from yt.data_objects.static_output import Dataset
from yt.funcs import mylog
from yt.geometry.grid_geometry_handler import GridIndex
from yt.utilities.on_demand_imports import _xarray as xr

from .fields import CFRadialFieldInfo


class CFRadialGrid(AMRGridPatch):
    _id_offset = 0

    def __init__(self, id, index, level, dimensions):
        super().__init__(id, filename=index.index_filename, index=index)
        self.Parent = None
        self.Children = []
        self.Level = level
        self.ActiveDimensions = dimensions

    def __repr__(self):
        return "CFRadialGrid_%04i (%s)" % (self.id, self.ActiveDimensions)


class CFRadialHierarchy(GridIndex):
    grid = CFRadialGrid

    def __init__(self, ds, dataset_type="cf_radial"):
        self.dataset_type = dataset_type
        self.dataset = weakref.proxy(ds)
        # our index file is the dataset itself:
        self.index_filename = self.dataset.parameter_filename
        self.directory = os.path.dirname(self.index_filename)
        # float type for the simulation edges and must be float64 now
        self.float_type = np.float64
        super().__init__(ds, dataset_type)

    def _initialize_state_variables(self):
        super()._initialize_state_variables()

    def _detect_output_fields(self):
        # This sets self.field_list, containing all the available on-disk fields and
        # records the units for each field.
        self.field_list = []
        units = {}
        for key in self.ds._handle.variables.keys():
            if (
                all(x in self.ds._handle[key].dims for x in ["time", "z", "y", "x"])
                is True
            ):
                fld = ("cf_radial", key)
                self.field_list.append(fld)
                units[fld] = self.ds._handle[key].units

        self.ds.field_units.update(units)

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
        # only a single grid, no need to loop
        g = self.grid(0, self, self.grid_levels.flat[0], self.grid_dimensions[0])
        g._prepare_grid()
        g._setup_dx()
        self.grids = np.array([g], dtype="object")


class CFRadialDataset(Dataset):
    _index_class = CFRadialHierarchy
    _field_info_class = CFRadialFieldInfo

    def __init__(
        self,
        filename,
        dataset_type="cf_radial",
        grid_shape=None,  # e.g., (nz, nx, ny)
        grid_limits=None,  # e.g., ((zmin, zmax), (xmin, xmax), (ymin, ymax))
        grid_kwargs=None,
        storage_filename=None,
        units_override=None,
    ):
        self.fluid_types += ("cf_radial",)
        self._handle = xr.open_dataset(filename)
        self.grid_shape = grid_shape
        self.grid_limits = grid_limits

        if "x" not in self._handle.coords:

            mylog.info(
                "Generating a cartesian grid for file %s. Data will be loaded "
                "from the cartesian grid.",
                filename,
            )

            grid_shape = self.grid_shape
            grid_limits = self.grid_limits
            if grid_shape is None or grid_limits is None:
                grid_shape, grid_limits = self._get_regrid_params()
                mylog.warning(
                    "grid_shape and grid_limits not supplied, using generated"
                    " values of %s and %s, respectively.",
                    grid_shape,
                    grid_limits,
                )

            from yt.utilities.on_demand_imports import _pyart as pyart

            radar = pyart.io.read_cfradial(filename)
            if grid_kwargs is None:
                grid_kwargs = {}
            grid = pyart.map.grid_from_radars(
                (radar,), grid_shape, grid_limits, **grid_kwargs
            )
            new_handle = grid.to_xarray()

            # attributes dont copy over, make sure we dont lose them
            new_handle.attrs.update(self._handle.attrs)
            # a few other attributes are stored as fields in the raw files
            for attr in ["latitude", "longitude", "altitude"]:
                new_handle.attrs["origin_" + attr] = self._handle[attr]
            self._handle = new_handle
            mylog.info(
                "Cartesian grid generation complete. To control the cartesian gridding, "
                "supply grid_limits and grid_shape arguments to yt.load()."
            )

        super().__init__(filename, dataset_type, units_override=units_override)
        self.refine_by = 2  # refinement factor between a grid and its subgrid

    def _set_code_unit_attributes(self):
        length_unit = self._handle.variables["x"].attrs["units"]
        self.length_unit = self.quan(1.0, length_unit)
        self.mass_unit = self.quan(1.0, "kg")
        self.time_unit = self.quan(1.0, "s")

    def _parse_parameter_file(self):
        self.parameters = {}

        x, y, z = [self._handle.coords[d] for d in "xyz"]

        self.origin_latitude = self._handle.origin_latitude
        self.origin_longitude = self._handle.origin_longitude

        self.domain_left_edge = np.array([x.min(), y.min(), z.min()])
        self.domain_right_edge = np.array([x.max(), y.max(), z.max()])
        self.dimensionality = 3
        dims = [self._handle.dims[d] for d in "xyz"]
        self.domain_dimensions = np.array(dims, dtype="int64")
        self._periodicity = (False, False, False)
        self.current_time = 0.0  # float(self._handle.time.values)

        # Cosmological information set to zero (not in space).
        self.cosmological_simulation = 0
        self.current_redshift = 0.0
        self.omega_lambda = 0.0
        self.omega_matter = 0.0
        self.hubble_constant = 0.0

    def _get_regrid_params(self, max_n=100):
        """
        calculates a cartesian bounding box covering the extents of a raw radar
        file and grid element size based on the raw data.

        Parameters
        ----------
        max_n : int
            max number of grid points along any dimension, ensures that the
            initial guess at a grid remains coarse enough for quick processing.
            User should supply their own grid_shape and grid_limits arguments
            to yt.load() or ds.grid_from_radars() to use a finer grid.

        Returns
        -------
        (grid_shape, grid_limits)
            grid_shape:  tuple, e.g., (40, 200, 200)
            grid_limits: tuple, e.g., ((0.0, 2000.0), (-1e5, 1e5), (-1e5, 1e5))
        """

        ds_xr = self._handle  # the xarray dataset handle
        has_crds = np.prod([hasattr(ds_xr, fld) for fld in ["range", "elevation"]])
        if has_crds:
            max_r = np.float64(ds_xr.range.max().values)  # range = radius
            if hasattr(ds_xr.range, "meters_between_gates"):
                dr = ds_xr.range.meters_between_gates
            else:
                r = ds_xr.range.values
                dr = np.mean(r[1:] - r[0:-1])

            # max z is at elevation max
            theta_max = np.float64(ds_xr.elevation.max().values) * np.pi / 180
            z_max = np.ceil(max_r * np.sin(theta_max))
            dz = np.ceil(dr * np.sin(theta_max))

            # max x/y is at elevation min
            theta_min = np.float64(ds_xr.elevation.min().values) * np.pi / 180
            xy_max = np.ceil(max_r * np.cos(theta_min))
            dxy = np.ceil(dr * np.cos(theta_min))

            dvals = (dz, dxy, dxy)
            grid_limits = ((0.0, z_max), (-xy_max, xy_max), (-xy_max, xy_max))
            grid_wid = [d[1] - d[0] for d in grid_limits]
            grid_shape = [int(wid / dv) for wid, dv in zip(grid_wid, dvals)]
            if max_n:
                # limit the number of elements along each dimensions, scaled
                # by the aspect ratio of the grid shape so we dont end up with
                # elongated grid elements
                z_xy = float(grid_shape[0]) / grid_shape[1]  # initial aspect ratio
                grid_shape = [N if N <= max_n else max_n for N in grid_shape]
                if z_xy < 1:
                    grid_shape = [g * asp for g, asp in zip(grid_shape, [z_xy, 1, 1])]
                else:
                    grid_shape = [
                        g * asp for g, asp in zip(grid_shape, [1, z_xy, z_xy])
                    ]
                grid_shape = tuple([int(g) for g in grid_shape])
            return grid_shape, grid_limits

        mylog.error(
            "Could not determine initial grid, please supply grid_shape "
            "and grid_limits arguments to yt.load()"
        )
        raise AttributeError

    @classmethod
    def _is_valid(cls, filename, *args, **kwargs):
        # This accepts a filename or a set of arguments and returns True or
        # False depending on if the file is of the type requested.

        try:
            ds = xr.open_dataset(filename)
        except (ImportError, OSError, AttributeError, TypeError):
            # catch all these to avoid errors when xarray cant handle a file
            return False

        if hasattr(ds, "attrs") and isinstance(ds.attrs, dict):
            con = "Conventions"
            return "CF/Radial" in ds.attrs.get(con, "") + ds.attrs.get(con.lower(), "")
        return False
