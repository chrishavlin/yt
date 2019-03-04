"""
Skeleton data structures



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

import os
import numpy as np
import weakref
import xarray

from yt.data_objects.grid_patch import \
    AMRGridPatch
from yt.geometry.grid_geometry_handler import \
    GridIndex
from yt.data_objects.static_output import \
    Dataset
from .fields import CM1FieldInfo


class CM1Grid(AMRGridPatch):
    _id_offset = 0

    def __init__(self, id, index, level):
        super(CM1Grid, self).__init__(
            id, filename=index.index_filename, index=index)
        self.Parent = None
        self.Children = []
        self.Level = level

    def __repr__(self):
        return "CM1Grid_%04i (%s)" % (self.id, self.ActiveDimensions)


class CM1Hierarchy(GridIndex):
    grid = CM1Grid

    def __init__(self, ds, dataset_type='cm1'):
        self.dataset_type = dataset_type
        self.dataset = weakref.proxy(ds)
        # for now, the index file is the dataset!
        self.index_filename = self.dataset.parameter_filename
        self.directory = os.path.dirname(self.index_filename)
        # float type for the simulation edges and must be float64 now
        self.float_type = np.float64
        super(CM1Hierarchy, self).__init__(ds, dataset_type)

    def _detect_output_fields(self):
        # This needs to set a self.field_list that contains all the available,
        # on-disk fields. No derived fields should be defined here.
        # NOTE: Each should be a tuple, where the first element is the on-disk
        # fluid type or particle type.  Convention suggests that the on-disk
        # fluid type is usually the dataset_type and the on-disk particle type
        # (for a single population of particles) is "io".
        pass

    def _count_grids(self):
        # This needs to set self.num_grids
        pass

    def _parse_index(self):
        # This needs to fill the following arrays, where N is self.num_grids:
        #   self.grid_left_edge         (N, 3) <= float64
        #   self.grid_right_edge        (N, 3) <= float64
        #   self.grid_dimensions        (N, 3) <= int
        #   self.grid_particle_count    (N, 1) <= int
        #   self.grid_levels            (N, 1) <= int
        #   self.grids                  (N, 1) <= grid objects
        #   self.max_level = self.grid_levels.max()
        pass

    def _populate_grid_objects(self):
        # For each grid g, this must call:
        #   g._prepare_grid()
        #   g._setup_dx()
        # This must also set:
        #   g.Children <= list of child grids
        #   g.Parent   <= parent grid
        # This is handled by the frontend because often the children must be
        # identified.
        pass


class CM1Dataset(Dataset):
    _index_class = CM1Hierarchy
    _field_info_class = CM1FieldInfo

    def __init__(self, filename, dataset_type='cm1',
                 storage_filename=None,
                 units_override=None):
        self.fluid_types += ('cm1',)
	self._handle = xarray.open_dataset(filename)
        # refinement factor between a grid and its subgrid
	self.refine_by = 2
        super(CM1Dataset, self).__init__(filename, dataset_type,
                         units_override=units_override)
        self.storage_filename = storage_filename

    def _set_code_unit_attributes(self):
        # This is where quantities are created that represent the various
        # on-disk units.  These are the currently available quantities which
        # should be set, along with examples of how to set them to standard
        # values.
        #
        # self.length_unit = self.quan(1.0, "cm")
        # self.mass_unit = self.quan(1.0, "g")
        # self.time_unit = self.quan(1.0, "s")
        # self.time_unit = self.quan(1.0, "s")
        #
        # These can also be set:
        # self.velocity_unit = self.quan(1.0, "cm/s")
        # self.magnetic_unit = self.quan(1.0, "gauss")
        length_unit = self._handle.variables['xh'].attrs['units']
        self.length_unit = self.quan(1.0, length_unit)
        self.mass_unit = self.quan(1.0, "kg")
        self.time_unit = self.quan(1.0, "s")
	self.velocity_unit = self.quan(1.0, "m/s")

    def _parse_parameter_file(self):
        # This needs to set up the following items.  Note that these are all
        # assumed to be in code units; domain_left_edge and domain_right_edge
        # will be converted to YTArray automatically at a later time.
        # This includes the cosmological parameters.
        #
        #   self.unique_identifier      <= unique identifier for the dataset
        #                                  being read (e.g., UUID or ST_CTIME)
	self.unique_identifier = int(os.stat(self.parameter_filename)[stat.ST_CTIME])
        #   self.parameters             <= full of code-specific items of use
	self.parameters = {}
	coords = self._handle.coords
	# TO DO: Possibly figure out a way to generalize this to be coordiante variable name
	# agnostic in order to make useful for WRF or climate data. For now, we're hard coding
	# for CM1 specifically and have named the classes appropriately, but generalizing is good.
	xh, yh, zh = [coords[i] for i in ["xh", "yh", "zh"]]
        #   self.domain_left_edge       <= array of float64
	self.domain_left_edge = np.array([xh.min(), yh.min(), zh.min()])
        #   self.domain_right_edge      <= array of float64
	self.domain_right_edge = np.array([xh.max(), yh.max(), zh.max()])
        #   self.dimensionality         <= int
	self.dimensionality = 3
        #   self.domain_dimensions      <= array of int64
	dims = [self._handle.dims[i] for i in ["xh", "yh", "zh"]]
	self.domain_dimensions = np.array(dims, dtype='int64')
        #   self.periodicity            <= three-element tuple of booleans
	self.periodicicity = (False, False, False)
        #   self.current_time           <= simulation time in code units
	self.current_time = self._handle.time.values
        # We also set up cosmological information.  Set these to zero if
        # non-cosmological.
	self.cosmological_simulation = 0.0
        self.current_redshift = 0.0
        self.omega_lambda = 0.0
        self.omega_matter = 0.0
        self.hubble_constant = 0.0

    @classmethod
    def _is_valid(self, *args, **kwargs):
        # This accepts a filename or a set of arguments and returns True or
        # False depending on if the file is of the type requested.
        try:
            import xarray
        except ImportError:
            return False

        try:
            ds = xarray.open_dataset(args[0])
        except (FileNotFoundError, OSError):
            return False

        try:
            variables = ds.variables.keys()
        except KeyError:
            return False

        if 'xh' in variables:
            return True

        return False
