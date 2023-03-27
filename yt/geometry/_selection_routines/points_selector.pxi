cdef class PointsSelector(SelectorObject):
    cdef double[:, :] p #np.ndarray[np.float64_t, ndim=2] p
    cdef int n_pts

    def __init__(self, dobj):

        cdef int i_pt, i_dim
        cdef np.float64_t[:] DLE = _ensure_code(dobj.ds.domain_left_edge)
        cdef np.float64_t[:] DRE = _ensure_code(dobj.ds.domain_right_edge)

        # loop over the points
        self.n_pts = dobj.p.shape[0]
        n_dims = dobj.p.shape[1]
        self.p = np.empty(dobj.p.shape, dtype=np.float64)
        for i_pt in range(self.n_pts):
            for i_dim in range(3):
                self.p[i_pt, i_dim] = _ensure_code(dobj.p[i_pt, i_dim])

                # ensure the point lies in the domain
                if self.periodicity[i_dim]:
                    self.p[i_pt, i_dim] = np.fmod(self.p[i_pt, i_dim], self.domain_width[i_dim])
                    if self.p[i_pt, i_dim] < DLE[i_dim]:
                        self.p[i_pt, i_dim] += self.domain_width[i_dim]
                    elif self.p[i_pt, i_dim] >= DRE[i_dim]:
                        self.p[i_pt, i_dim] -= self.domain_width[i_dim]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_cell(self, np.float64_t pos[3], np.float64_t dds[3]) nogil:

        cdef int i_pt
        for i_pt in range(self.n_pts):
            if (pos[0] - 0.5*dds[0] <= self.p[i_pt, 0] < pos[0]+0.5*dds[0] and
                pos[1] - 0.5*dds[1] <= self.p[i_pt, 1] < pos[1]+0.5*dds[1] and
                pos[2] - 0.5*dds[2] <= self.p[i_pt, 2] < pos[2]+0.5*dds[2]):
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
        # hmm not sure about the hash
        return (("p[0]", self.p[:,0]),
                ("p[1]", self.p[:,1]),
                ("p[2]", self.p[:,2]))

    def _get_state_attnames(self):
        return ('pts', )

points_selector = PointsSelector
