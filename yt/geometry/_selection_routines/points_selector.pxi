@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef init_points(double[:,:] output_p, np.float64_t[:, :] input_p):
    cdef int n_pts = input_p.shape[0]
    cdef int n_dim = input_p.shape[1]
    for i_pt in range(n_pts):
           for i_dim in range(n_dim):
                output_p[i_pt, i_dim] = input_p[i_pt, i_dim]


cdef class PointsSelector(SelectorObject):
    cdef double[:, :] p
    cdef int n_pts

    def __init__(self, dobj):

        cdef int i_pt, i_dim
        cdef np.float64_t[:] DLE = _ensure_code(dobj.ds.domain_left_edge)
        cdef np.float64_t[:] DRE = _ensure_code(dobj.ds.domain_right_edge)

        # loop over the points
        self.n_pts = dobj.p.shape[0]
        n_dims = dobj.p.shape[1]
        self.p = np.empty(dobj.p.shape, dtype=np.float64)
        init_points(self.p, dobj.p)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_cell(self, np.float64_t pos[3], np.float64_t dds[3]) nogil:

        cdef int i_pt
        cdef np.float64_t p0m = pos[0] - 0.5*dds[0]
        cdef np.float64_t p0p = pos[0] + 0.5*dds[0]
        cdef np.float64_t p1m = pos[1] - 0.5*dds[1]
        cdef np.float64_t p1p = pos[1] + 0.5*dds[1]
        cdef np.float64_t p2m = pos[2] - 0.5*dds[2]
        cdef np.float64_t p2p = pos[2] + 0.5*dds[2]

        for i_pt in range(self.n_pts):
            if (p0m <= self.p[i_pt, 0] < p0p and
                p1m <= self.p[i_pt, 1] < p1p and
                p2m <= self.p[i_pt, 2] < p2p):
                return 1

        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_sphere(self, np.float64_t pos[3], np.float64_t radius) nogil:

        cdef int i_pt
        cdef int i
        cdef np.float64_t dist, dist2 = 0

        for i_pt in range(self.n_pts):
            for i in range(3):
                dist = self.periodic_difference(pos[i], self.p[i_pt, i], i)
                dist2 += dist*dist
            if dist2 <= radius*radius: return 1
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_bbox(self, np.float64_t left_edge[3],
                               np.float64_t right_edge[3]) nogil:
        cdef int i_pt
        for i_pt in range(self.n_pts):
            # point definitely can only be in one cell
            if (left_edge[0] <= self.p[i_pt, 0] < right_edge[0] and
                left_edge[1] <= self.p[i_pt, 1] < right_edge[1] and
                left_edge[2] <= self.p[i_pt, 2] < right_edge[2]):
                return 1
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_bbox_edge(self, np.float64_t left_edge[3],
                               np.float64_t right_edge[3]) nogil:

        cdef int i_pt
        for i_pt in range(self.n_pts):
            # point definitely can only be in one cell
            # Return 2 in all cases to indicate that the point only overlaps
            # portion of box
            if (left_edge[0] <= self.p[i_pt, 0] <= right_edge[0] and
                left_edge[1] <= self.p[i_pt, 1] <= right_edge[1] and
                left_edge[2] <= self.p[i_pt, 2] <= right_edge[2]):
                return 2

        return 0

    def _hash_vals(self):
        # this should be changed.
        # currently it will read from cache if just the first point matches.
        return (("p[0]", self.p[:,0]),
                ("p[1]", self.p[:,1]),
                ("p[2]", self.p[:,2]))

    def _get_state_attnames(self):
        return ('pts', )

points_selector = PointsSelector
