from libc.math cimport sin, cos, atan2, sqrt, acos

cdef class CuttingPlaneSelector(SelectorObject):
    cdef public np.float64_t norm_vec[3]  # the unit-normal for the plane
    cdef public np.float64_t d  # the shortest distance from plane to origin

    def __init__(self, dobj):
        cdef int i
        for i in range(3):
            self.norm_vec[i] = dobj._norm_vec[i]
        self.d = _ensure_code(dobj._d)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_cell(self, np.float64_t pos[3], np.float64_t dds[3]) noexcept nogil:
        cdef np.float64_t left_edge[3]
        cdef np.float64_t right_edge[3]
        cdef int i
        for i in range(3):
            left_edge[i] = pos[i] - 0.5*dds[i]
            right_edge[i] = pos[i] + 0.5*dds[i]
        return self.select_bbox(left_edge, right_edge)

    cdef int select_point(self, np.float64_t pos[3]) noexcept nogil:
        # two 0-volume constructs don't intersect
        return 0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_sphere(self, np.float64_t pos[3], np.float64_t radius) noexcept nogil:
        cdef int i
        cdef np.float64_t height = self.d
        for i in range(3) :
            height += pos[i] * self.norm_vec[i]
        if height*height <= radius*radius : return 1
        return 0


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void transform_vertex_pos(self, np.float64_t pos_in[3], np.float64_t pos_out[3]) noexcept nogil:
        pass

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_bbox(self, np.float64_t left_edge[3],
                               np.float64_t right_edge[3]) noexcept nogil:
        return self._select_bbox(left_edge, right_edge)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int _select_bbox(self, np.float64_t left_edge[3],
                               np.float64_t right_edge[3]) noexcept nogil:
        # the bbox selection here works by calculating the signed-distance from
        # the plane to each vertex of the bounding box. If there is no
        # intersection, the signed-distance for every vertex will have the same
        # sign whereas if the sign flips then the plane must intersect the
        # bounding box.
        cdef int i, j, k, n
        cdef np.float64_t *arr[2]
        cdef np.float64_t pos[3]
        cdef np.float64_t gd
        arr[0] = left_edge
        arr[1] = right_edge
        all_under = 1
        all_over = 1
        # Check each corner
        for i in range(2):
            pos[0] = arr[i][0]
            for j in range(2):
                pos[1] = arr[j][1]
                for k in range(2):
                    pos[2] = arr[k][2]
                    self.transform_vertex_pos(pos, pos)
                    gd = self.d
                    for n in range(3):
                        gd += pos[n] * self.norm_vec[n]
                    # this allows corners and faces on the low-end to
                    # collide, while not selecting cells on the high-side
                    if i == 0 and j == 0 and k == 0 :
                        if gd <= 0: all_over = 0
                        if gd >= 0: all_under = 0
                    else :
                        if gd < 0: all_over = 0
                        if gd > 0: all_under = 0
        if all_over == 1 or all_under == 1:
            return 0
        return 1


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_bbox_edge(self, np.float64_t left_edge[3],
                               np.float64_t right_edge[3]) noexcept nogil:
        cdef int i, j, k, n
        cdef np.float64_t *arr[2]
        cdef np.float64_t pos[3]
        cdef np.float64_t gd
        arr[0] = left_edge
        arr[1] = right_edge
        all_under = 1
        all_over = 1
        # Check each corner
        for i in range(2):
            pos[0] = arr[i][0]
            for j in range(2):
                pos[1] = arr[j][1]
                for k in range(2):
                    pos[2] = arr[k][2]
                    gd = self.d
                    for n in range(3):
                        gd += pos[n] * self.norm_vec[n]
                    # this allows corners and faces on the low-end to
                    # collide, while not selecting cells on the high-side
                    if i == 0 and j == 0 and k == 0 :
                        if gd <= 0: all_over = 0
                        if gd >= 0: all_under = 0
                    else :
                        if gd < 0: all_over = 0
                        if gd > 0: all_under = 0
        if all_over == 1 or all_under == 1:
            return 0
        return 2 # a box of non-zeros volume can't be inside a plane

    def _hash_vals(self):
        return (("norm_vec[0]", self.norm_vec[0]),
                ("norm_vec[1]", self.norm_vec[1]),
                ("norm_vec[2]", self.norm_vec[2]),
                ("d", self.d))

    def _get_state_attnames(self):
        return ("d", "norm_vec")

cdef class SphericalCuttingPlaneSelector(CuttingPlaneSelector):

    # interesection of a cartesian plane with data in spherical coordinates.
    # expected ordering is (r, theta, phi), where theta is the azimuthal/latitudinal
    # angle (bounds of 0 to pi) and phi is the polar/longitudinal angle (bounds 0 to 2pi).

    cdef public np.float64_t r_min # the minimum radius for possible intersection
    cdef public np.float64_t c_rtp[3]
    cdef public np.float64_t c_xyz[3]

    def __init__(self, dobj):

        cdef np.float64_t xyz[3]
        cdef int i
        super().__init__(dobj)

        # any points at r < |d|, where d is the minimum distance-vector to the
        # plane, cannot intersect the plane. Record r_min here for convenience:
        self.r_min = fabs(self.d)

        # also record the phi and theta coordinates of the point on the plane
        # closest to the origin
        for i in range(3):
            xyz[i] = - self.norm_vec[i] * self.d  # cartesian position
            self.c_xyz[i] = xyz[i] # temp for debugging
        self.transform_xyz_to_rtp(xyz, self.c_rtp)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void transform_rtp_to_xyz(self, np.float64_t in_pos[3], np.float64_t out_pos[3]) noexcept nogil:
        # See `spherical_coordinates.py` for more details
        # r => in_pos[0] theta => in_pos[1] phi => in_pos[2]
        out_pos[0] = in_pos[0] * cos(in_pos[2]) * sin(in_pos[1])
        out_pos[1] = in_pos[0] * sin(in_pos[2]) * sin(in_pos[1])
        out_pos[2] = in_pos[0] * cos(in_pos[1])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void transform_vertex_pos(self, np.float64_t pos_in[3], np.float64_t pos_out[3]) noexcept nogil:
        self.transform_rtp_to_xyz(pos_in, pos_out)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef void transform_xyz_to_rtp(self, np.float64_t in_pos[3], np.float64_t out_pos[3]) noexcept nogil:
        # we have to have 012 be xyz and 012 be rtp
        out_pos[0] = sqrt(in_pos[0]*in_pos[0] + in_pos[1]*in_pos[1] + in_pos[2]*in_pos[2])
        out_pos[1] = acos(in_pos[2] / out_pos[0])
        out_pos[2] = atan2(in_pos[1], in_pos[0])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef int select_bbox(self, np.float64_t left_edge[3],
                               np.float64_t right_edge[3]) noexcept nogil:
         # left/right edge here are in spherical coordinates in (r, theta, phi)

         cdef int selected
         cdef np.float64_t left_edge_c[3], right_edge_c[3]

         # first check closest-approach condition
         if right_edge[0] <= self.r_min:
             # intersection impossible!
             return 0

         # run the plane-vertex distance check (vertex positions are converted to 
         # cartesian within _select_bbox
         selected = self._select_bbox(left_edge, right_edge)

         if selected == 0:
              # there is one special case to consider!
              # When all vertices lie on one side of the plane, intersection
              # is still possible if the plane intersects the outer cusp of the
              # spherical volume element. **BUT** if we've reached this far,
              # the only way for this to happen is if the position of the point
              # on the plane that is closest to the origin lies within the
              # element itself.
              if self.c_rtp[0] > left_edge[0]:
                  for idim in range(1,3):
                      if self.c_rtp[idim] <= left_edge[idim]:
                          return 0
                      if self.c_rtp[idim] >= right_edge[idim]:
                          return 0
                  return 1


         return selected

    def _select_bbox_temp(self,
                  left_edge_in,
                  right_edge_in):

         # useful for direct testing without having to initialize
         # full yt data objects

         cdef np.float64_t left_edge[3]
         cdef np.float64_t right_edge[3]

         for i in range(3):
             left_edge[i] = left_edge_in[i]
             right_edge[i] = right_edge_in[i]

         return self.select_bbox(left_edge, right_edge)

    def check_edge_conversion(self, left_edge_in, right_edge_in):
        # debugging, delete eventually

        cdef np.float64_t left_edge[3]
        cdef np.float64_t right_edge[3]
        cdef np.float64_t left_edge_c[3]
        cdef np.float64_t right_edge_c[3]

        for i in range(3):
            left_edge[i] = left_edge_in[i]
            right_edge[i] = right_edge_in[i]

        self.transform_rtp_to_xyz(left_edge, left_edge_c)
        self.transform_rtp_to_xyz(right_edge, right_edge_c)
        print(left_edge_c)
        print(right_edge_c)



cutting_selector = CuttingPlaneSelector

spherical_cutting_selector = SphericalCuttingPlaneSelector


