from yt.geometry.selection_routines import spherical_cutting_selector
import numpy as np


class HelpfulPlaneObject:
    # a bare-bones skeleton of a data object to use for initializing
    # the cython cutting plane selector
    def __init__(self, normal, plane_center):
        self._d = -1 * np.dot(normal, plane_center)
        self._norm_vec = normal


def test_spherical_cutting_plane():

    # define plane and initialize selector
    normal = np.array([0., 0., 1.])
    plane_center = np.array([0., 0., 1.])
    plane = HelpfulPlaneObject(normal, plane_center)
    scp = spherical_cutting_selector(plane)
    assert scp.r_min == 1.0
    print(scp.c_xyz)
    print(scp.c_rtp)
    
    # left/right edge values are given in spherical coordinates with
    # order of (r, theta, phi) where
    #   theta is the azimuthal/latitudinal
    #   phi is the polar/longitudinal angle (bounds 0 to 2pi).
    
    def _in_rads(x):
        return x*np.pi/180
    
    left_edge = np.array([0.8, _in_rads(5), _in_rads(5)])
    right_edge = np.array([1.2, _in_rads(45), _in_rads(45)])
    
    assert scp._select_bbox_temp(left_edge, right_edge)
