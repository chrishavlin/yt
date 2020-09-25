import numpy as np

from yt.utilities.file_handler import NetCDF4FileHandler
from yt.utilities.io_handler import BaseIOHandler


class IRISIOHandler(BaseIOHandler):
    _particle_reader = False
    _dataset_type = "IRIS"

    def __init__(self, ds):
        self.filename = ds.filename
        self._handle = NetCDF4FileHandler(self.filename)
        super(IRISIOHandler, self).__init__(ds)

    def _read_particle_coords(self, chunks, ptf):
        # This needs to *yield* a series of tuples of (ptype, (x, y, z)).
        # chunks is a list of chunks, and ptf is a dict where the keys are
        # ptypes and the values are lists of fields.
        pass

    def _read_particle_fields(self, chunks, ptf, selector):
        # This gets called after the arrays have been allocated.  It needs to
        # yield ((ptype, field), data) where data is the masked results of
        # reading ptype, field and applying the selector to the data read in.
        # Selector objects have a .select_points(x,y,z) that returns a mask, so
        # you need to do your masking here.
        pass

    def _read_fluid_selection(self, chunks, selector, fields, size):

        data = {}
        offset = 0
        with self._handle.open_ds() as ds:
            for field in fields:
                data[field] = np.empty(size, dtype="float64")
                for chunk in chunks:
                    for grid in chunk.objs:
                        variable = ds.variables[field[1]][:][0]
                        values = np.squeeze(variable.T)
                        offset += grid.select(selector, values, data[field], offset)
        return data

    def _read_chunk_data(self, chunk, fields):
        # This reads the data from a single chunk without doing any selection,
        # and is only used for caching data that might be used by multiple
        # different selectors later. For instance, this can speed up ghost zone
        # computation.
        pass
