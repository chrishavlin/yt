cimport numpy as np


cdef class MixedCoordBBox:
    cdef void get_cartesian_bbox(self,
                                np.float64_t pos0,
                                np.float64_t pos1,
                                np.float64_t pos2,
                                np.float64_t dpos0,
                                np.float64_t dpos1,
                                np.float64_t dpos2,
                                np.float64_t xyz_i[3],
                                np.float64_t dxyz_i[3]
                                ) noexcept nogil


cdef class SphericalMixedCoordBBox(MixedCoordBBox):
    cdef void get_cartesian_bbox(
                        self,
                        np.float64_t pos0,
                        np.float64_t pos1,
                        np.float64_t pos2,
                        np.float64_t dpos0,
                        np.float64_t dpos1,
                        np.float64_t dpos2,
                        np.float64_t xyz_i[3],
                        np.float64_t dxyz_i[3]
                        ) noexcept nogil


cdef class CartesianMixedCoordBBox(MixedCoordBBox):
    cdef void get_cartesian_bbox(
                        self,
                        np.float64_t pos0,
                        np.float64_t pos1,
                        np.float64_t pos2,
                        np.float64_t dpos0,
                        np.float64_t dpos1,
                        np.float64_t dpos2,
                        np.float64_t xyz_i[3],
                        np.float64_t dxyz_i[3]
                        ) noexcept nogil
