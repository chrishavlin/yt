import os
from collections import defaultdict

import numpy as np

from yt.frontends.sph.io import IOHandlerSPH
from yt.units.yt_array import uconcatenate
from yt.utilities.lib.particle_kdtree_tools import generate_smoothing_length
from yt.utilities.logger import ytLogger as mylog
from yt.utilities.on_demand_imports import _h5py as h5py

from .definitions import gadget_hdf5_ptypes


class IOHandlerGadgetHDF5(IOHandlerSPH):
    _dataset_type = "gadget_hdf5"
    _vector_fields = ("Coordinates", "Velocity", "Velocities", "MagneticField")
    _known_ptypes = gadget_hdf5_ptypes
    _var_mass = None
    _element_names = (
        "Hydrogen",
        "Helium",
        "Carbon",
        "Nitrogen",
        "Oxygen",
        "Neon",
        "Magnesium",
        "Silicon",
        "Iron",
    )

    _coord_name = "Coordinates"

    @property
    def var_mass(self):
        if self._var_mass is None:
            vm = []
            for i, v in enumerate(self.ds["Massarr"]):
                if v == 0:
                    vm.append(self._known_ptypes[i])
            self._var_mass = tuple(vm)
        return self._var_mass

    def _read_fluid_selection(self, chunks, selector, fields, size):
        raise NotImplementedError

    def _read_particle_coords(self, chunks, ptf):
        for data_file in self._sorted_chunk_iterator(chunks):
            with data_file.transaction as handle:
                # This double-reads
                for ptype in sorted(ptf):
                    if data_file.total_particles[ptype] == 0:
                        continue
                    c = data_file._read_field(ptype, self._coord_name, handle).astype(
                        "f8"
                    )
                    x, y, z = (np.squeeze(_) for _ in np.split(c, 3, axis=1))
                    if ptype == self.ds._sph_ptypes[0]:
                        pdtype = c.dtype
                        pshape = c.shape
                        hsml = self._get_smoothing_length(data_file, pdtype, pshape)
                    else:
                        hsml = 0.0
                    yield ptype, (x, y, z), hsml

    def _yield_coordinates(self, data_file, needed_ptype=None):
        si, ei = data_file.start, data_file.end
        f = h5py.File(data_file.filename, mode="r")
        pcount = f["/Header"].attrs["NumPart_ThisFile"][:].astype("int")
        np.clip(pcount - si, 0, ei - si, out=pcount)
        pcount = pcount.sum()
        for key in f.keys():
            if not key.startswith("PartType"):
                continue
            if "Coordinates" not in f[key]:
                continue
            if needed_ptype and key != needed_ptype:
                continue
            ds = f[key]["Coordinates"][si:ei, ...]
            dt = ds.dtype.newbyteorder("N")  # Native
            pos = np.empty(ds.shape, dtype=dt)
            pos[:] = ds
            yield key, pos
        f.close()

    def _generate_smoothing_length(self, index):
        data_files = index.data_files
        if not self.ds.gen_hsmls:
            return
        hsml_fn = data_files[0].filename.replace(".hdf5", ".hsml.hdf5")
        if os.path.exists(hsml_fn):
            with h5py.File(hsml_fn, mode="r") as f:
                file_hash = f.attrs["q"]
            if file_hash != self.ds._file_hash:
                mylog.warning("Replacing hsml files.")
                for data_file in data_files:
                    hfn = data_file.filename.replace(".hdf5", ".hsml.hdf5")
                    os.remove(hfn)
            else:
                return
        positions = []
        counts = defaultdict(int)
        for data_file in data_files:
            for _, ppos in self._yield_coordinates(
                data_file, needed_ptype=self.ds._sph_ptypes[0]
            ):
                counts[data_file.filename] += ppos.shape[0]
                positions.append(ppos)
        if not positions:
            return
        offsets = {}
        offset = 0
        for fn, count in counts.items():
            offsets[fn] = offset
            offset += count
        kdtree = index.kdtree
        positions = uconcatenate(positions)[kdtree.idx]
        hsml = generate_smoothing_length(
            positions.astype("float64"), kdtree, self.ds._num_neighbors
        )
        dtype = positions.dtype
        hsml = hsml[np.argsort(kdtree.idx)].astype(dtype)
        mylog.warning("Writing smoothing lengths to hsml files.")
        for i, data_file in enumerate(data_files):
            si, ei = data_file.start, data_file.end
            fn = data_file.filename
            hsml_fn = data_file.filename.replace(".hdf5", ".hsml.hdf5")
            with h5py.File(hsml_fn, mode="a") as f:
                if i == 0:
                    f.attrs["q"] = self.ds._file_hash
                g = f.require_group(self.ds._sph_ptypes[0])
                d = g.require_dataset(
                    "SmoothingLength", dtype=dtype, shape=(counts[fn],)
                )
                begin = si + offsets[fn]
                end = min(ei, d.size) + offsets[fn]
                d[si:ei] = hsml[begin:end]

    def _get_smoothing_length(
        self, data_file, position_dtype, position_shape, handle=None
    ):
        ptype = self.ds._sph_ptypes[0]
        if self.ds.gen_hsmls:
            fn = data_file.filename.replace(".hdf5", ".hsml.hdf5")
            si, ei = data_file.start, data_file.end
            with h5py.File(fn, mode="r") as f:
                ds = f[ptype]["SmoothingLength"][si:ei, ...]
        else:
            # note: if handle is None, _read_file will open it.
            ds = data_file._read_field(ptype, "SmoothingLength", handle)
            # note, previously, this method always opened the hdf file and
            # read f[ptype]["SmoothingLength"] without checking if there
            # are any particles of ptype. _read_field does include that check
            # and returns None if there are no particles, so we need to
            # return an empty array here:
            if ds is None:
                return np.empty(0, dtype=position_dtype)

        dt = ds.dtype.newbyteorder("N")  # Native
        if position_dtype is not None and dt < position_dtype:
            # Sometimes positions are stored in double precision
            # but smoothing lengths are stored in single precision.
            # In these cases upcast smoothing length to double precision
            # to avoid ValueErrors when we pass these arrays to Cython.
            dt = position_dtype
        hsml = np.empty(ds.shape, dtype=dt)
        hsml[:] = ds
        return hsml

    def _read_particle_data_file(self, data_file, ptf, selector=None):

        data_return = {}
        cname = self.ds._particle_coordinates_name
        with data_file.transaction as handle:

            for ptype, field_list in sorted(ptf.items()):
                if data_file.total_particles[ptype] == 0:
                    continue

                if selector is None or getattr(selector, "is_all_data", False):
                    mask = slice(None, None, None)
                    mask_sum = data_file.total_particles[ptype]
                    hsmls = None
                else:
                    coords = data_file._read_field(ptype, cname, handle).astype(
                        "float64"
                    )
                    if ptype == "PartType0":
                        hsmls = self._get_smoothing_length(
                            data_file, coords.dtype, coords.shape, handle
                        ).astype("float64")
                    else:
                        hsmls = 0.0
                    mask = selector.select_points(
                        coords[:, 0], coords[:, 1], coords[:, 2], hsmls
                    )
                    if mask is not None:
                        mask_sum = mask.sum()
                    del coords
                if mask is None:
                    continue

                for field in field_list:
                    if field in ("Mass", "Masses") and ptype not in self.var_mass:
                        data = np.empty(mask_sum, dtype="float64")
                        ind = self._known_ptypes.index(ptype)
                        data[:] = self.ds["Massarr"][ind]
                    elif field == "smoothing_length":
                        # This is for frontends which do not store
                        # the smoothing length on-disk, so we do not
                        # attempt to read them, but instead assume
                        # that they are calculated in _get_smoothing_length.
                        if hsmls is None:
                            coords = data_file._read_field(ptype, cname, handle).astype(
                                "float64"
                            )
                            hsmls = self._get_smoothing_length(
                                data_file,
                                coords.dtype,
                                coords.shape,
                                handle,
                            ).astype("float64")
                        data = hsmls[mask]
                    else:
                        data = data_file._read_field(ptype, field, handle)[mask]

                    data_return[(ptype, field)] = data
        return data_return

    def _count_particles(self, data_file):
        # temporary?
        return data_file._count_particles()

    def _identify_fields(self, data_file):
        f = h5py.File(data_file.filename, mode="r")
        fields = []
        cname = self.ds._particle_coordinates_name  # Coordinates
        mname = self.ds._particle_mass_name  # Mass

        # loop over all keys in OWLS hdf5 file
        # --------------------------------------------------
        for key in f.keys():

            # only want particle data
            # --------------------------------------
            if not key.startswith("PartType"):
                continue

            # particle data group
            # --------------------------------------
            g = f[key]
            if cname not in g:
                continue

            # note str => not unicode!
            ptype = str(key)
            if ptype not in self.var_mass:
                fields.append((ptype, mname))

            # loop over all keys in PartTypeX group
            # ----------------------------------------
            for k in g.keys():

                if k == "ElementAbundance":
                    gp = g[k]
                    for j in gp.keys():
                        kk = j
                        fields.append((ptype, str(kk)))
                elif (
                    k in ["Metallicity", "GFM_Metals", "PassiveScalars"]
                    and len(g[k].shape) > 1
                ):
                    # Vector of metallicity or passive scalar
                    for i in range(g[k].shape[1]):
                        fields.append((ptype, "%s_%02i" % (k, i)))
                elif k == "ChemistryAbundances" and len(g[k].shape) > 1:
                    for i in range(g[k].shape[1]):
                        fields.append((ptype, "Chemistry_%03i" % i))
                else:
                    kk = k
                    if not hasattr(g[kk], "shape"):
                        continue
                    if len(g[kk].shape) > 1:
                        self._vector_fields[kk] = g[kk].shape[1]
                    fields.append((ptype, str(kk)))

        f.close()

        if self.ds.gen_hsmls:
            fields.append(("PartType0", "smoothing_length"))

        return fields, {}


ZeroMass = object()


class IOHandlerGadgetBinary(IOHandlerSPH):
    _dataset_type = "gadget_binary"
    _vector_fields = (
        ("Coordinates", 3),
        ("Velocity", 3),
        ("Velocities", 3),
        ("MagneticField", 3),
        ("FourMetalFractions", 4),
        ("ElevenMetalMasses", 11),
    )

    # Particle types (Table 3 in GADGET-2 user guide)
    #
    # Blocks in the file:
    #   HEAD
    #   POS
    #   VEL
    #   ID
    #   MASS    (variable mass only)
    #   U       (gas only)
    #   RHO     (gas only)
    #   HSML    (gas only)
    #   POT     (only if enabled in makefile)
    #   ACCE    (only if enabled in makefile)
    #   ENDT    (only if enabled in makefile)
    #   TSTP    (only if enabled in makefile)

    _var_mass = None
    _format = None

    def __init__(self, ds, *args, **kwargs):
        self._vector_fields = dict(self._vector_fields)
        self._fields = ds._field_spec
        self._ptypes = ds._ptype_spec
        self.data_files = set()
        gformat, endianswap = ds._header.gadget_format
        # gadget format 1 original, 2 with block name
        self._format = gformat
        self._endian = endianswap
        super().__init__(ds, *args, **kwargs)

    @property
    def var_mass(self):
        if self._var_mass is None:
            vm = []
            for i, v in enumerate(self.ds["Massarr"]):
                if v == 0:
                    vm.append(self._ptypes[i])
            self._var_mass = tuple(vm)
        return self._var_mass

    def _read_fluid_selection(self, chunks, selector, fields, size):
        raise NotImplementedError

    def _read_particle_coords(self, chunks, ptf):
        data_files = set()
        for chunk in chunks:
            for obj in chunk.objs:
                data_files.update(obj.data_files)
        for data_file in sorted(data_files, key=lambda x: (x.filename, x.start)):
            poff = data_file.field_offsets
            tp = data_file.total_particles
            f = open(data_file.filename, "rb")
            for ptype in ptf:
                if tp[ptype] == 0:
                    # skip if there are no particles
                    continue
                f.seek(poff[ptype, "Coordinates"], os.SEEK_SET)
                pos = self._read_field_from_file(f, tp[ptype], "Coordinates")
                if ptype == self.ds._sph_ptypes[0]:
                    f.seek(poff[ptype, "SmoothingLength"], os.SEEK_SET)
                    hsml = self._read_field_from_file(f, tp[ptype], "SmoothingLength")
                else:
                    hsml = 0.0
                yield ptype, (pos[:, 0], pos[:, 1], pos[:, 2]), hsml
            f.close()

    def _read_particle_data_file(self, data_file, ptf, selector=None):
        return_data = {}
        poff = data_file.field_offsets
        tp = data_file.total_particles
        f = open(data_file.filename, "rb")
        for ptype, field_list in sorted(ptf.items()):
            if tp[ptype] == 0:
                continue
            if selector is None or getattr(selector, "is_all_data", False):
                mask = slice(None, None, None)
            else:
                f.seek(poff[ptype, "Coordinates"], os.SEEK_SET)
                pos = self._read_field_from_file(f, tp[ptype], "Coordinates")
                if ptype == self.ds._sph_ptypes[0]:
                    f.seek(poff[ptype, "SmoothingLength"], os.SEEK_SET)
                    hsml = self._read_field_from_file(f, tp[ptype], "SmoothingLength")
                else:
                    hsml = 0.0
                mask = selector.select_points(pos[:, 0], pos[:, 1], pos[:, 2], hsml)
                del pos
                del hsml
            if mask is None:
                continue
            for field in field_list:
                if field == "Mass" and ptype not in self.var_mass:
                    if getattr(selector, "is_all_data", False):
                        size = data_file.total_particles[ptype]
                    else:
                        size = mask.sum()
                    data = np.empty(size, dtype="float64")
                    m = self.ds.parameters["Massarr"][self._ptypes.index(ptype)]
                    data[:] = m
                    yield (ptype, field), data
                    continue
                f.seek(poff[ptype, field], os.SEEK_SET)
                data = self._read_field_from_file(f, tp[ptype], field)
                data = data[mask, ...]
                return_data[(ptype, field)] = data
        f.close()
        return return_data

    def _read_field_from_file(self, f, count, name):
        if count == 0:
            return
        if name == "ParticleIDs":
            dt = self._endian + self.ds._id_dtype
        else:
            dt = self._endian + self._float_type
        dt = np.dtype(dt)
        if name in self._vector_fields:
            count *= self._vector_fields[name]
        arr = np.fromfile(f, dtype=dt, count=count)
        # ensure data are in native endianness to avoid errors
        # when field data are passed to cython
        dt = dt.newbyteorder("N")
        arr = arr.astype(dt)
        if name in self._vector_fields:
            factor = self._vector_fields[name]
            arr = arr.reshape((count // factor, factor), order="C")
        return arr

    def _yield_coordinates(self, data_file, needed_ptype=None):
        self._float_type = data_file._header.float_type
        self._field_size = np.dtype(self._float_type).itemsize
        with open(data_file.filename, "rb") as f:
            # We add on an additionally 4 for the first record.
            f.seek(data_file._position_offset + 4)
            for ptype, count in data_file.total_particles.items():
                if count == 0:
                    continue
                if needed_ptype is not None and ptype != needed_ptype:
                    continue
                # The first total_particles * 3 values are positions
                pp = np.fromfile(f, dtype=self._float_type, count=count * 3)
                pp.shape = (count, 3)
                yield ptype, pp

    def _get_smoothing_length(self, data_file, position_dtype, position_shape):
        ret = self._get_field(data_file, "SmoothingLength", "Gas")
        if position_dtype is not None and ret.dtype != position_dtype:
            # Sometimes positions are stored in double precision
            # but smoothing lengths are stored in single precision.
            # In these cases upcast smoothing length to double precision
            # to avoid ValueErrors when we pass these arrays to Cython.
            ret = ret.astype(position_dtype)
        return ret

    def _get_field(self, data_file, field, ptype):
        poff = data_file.field_offsets
        tp = data_file.total_particles
        with open(data_file.filename, "rb") as f:
            f.seek(poff[ptype, field], os.SEEK_SET)
            pp = self._read_field_from_file(f, tp[ptype], field)
        return pp

    def _count_particles(self, data_file):
        return data_file._count_particles()

    # header is 256, but we have 4 at beginning and end for ints
    _field_size = 4

    def _identify_fields(self, domain):
        # We can just look at the particle counts.
        field_list = []
        tp = domain.total_particles
        for i, ptype in enumerate(self._ptypes):
            count = tp[ptype]
            if count == 0:
                continue
            m = domain.header["Massarr"][i]
            for field in self._fields:
                if isinstance(field, tuple):
                    field, req = field
                    if req is ZeroMass:
                        if m > 0.0:
                            continue
                    elif isinstance(req, tuple) and ptype in req:
                        pass
                    elif req != ptype:
                        continue
                field_list.append((ptype, field))
        return field_list, {}
