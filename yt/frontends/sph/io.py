"""
Generic file-handing functions for SPH data




"""
from dask import compute, delayed

from yt.utilities.io_handler import BaseIOHandler


class IOHandlerSPH(BaseIOHandler):
    """IOHandler implementation specifically for SPH data

    This exists to handle particles with smoothing lengths, which require us
    to read in smoothing lengths along with the the particle coordinates to
    determine particle extents.
    """

    def _count_particles_chunks(self, psize, chunks, ptf, selector):
        if getattr(selector, "is_all_data", False):
            chunks = list(chunks)
            data_files = set()
            for chunk in chunks:
                for obj in chunk.objs:
                    data_files.update(obj.data_files)
            data_files = sorted(data_files, key=lambda x: (x.filename, x.start))
            for data_file in data_files:
                for ptype in ptf.keys():
                    psize[ptype] += data_file.total_particles[ptype]
        else:
            for ptype, (x, y, z), hsml in self._read_particle_coords(chunks, ptf):
                psize[ptype] += selector.count_points(x, y, z, hsml)
        return dict(psize)

    def _count_particles_by_chunk(self, chunks, ptf, selector):
        # returns a list of psize dicts, one for each chunk, for setting
        # dask array chunk sizes. For now, it always applies the selector
        # as chunk-ordering is important.
        dlayd = [delayed(self._count_chunk_points)(ch, ptf, selector) for ch in chunks]
        psize_by_chunk = compute(*dlayd)  # sizes by chunk

        return psize_by_chunk

    def _count_chunk_points(self, chunk, ptf, selector):
        # returns particle size dict for a single chunk
        chunk_psize = {}
        for ptype, (x, y, z), hsml in self._read_particle_coords([chunk], ptf):
            chunk_psize[ptype] = selector.count_points(x, y, z, hsml)

        return chunk_psize
