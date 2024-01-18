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

    # left/right edge values are given in spherical coordinates with
    # order of (r, theta, phi) where
    #   theta is the azimuthal/latitudinal
    #   phi is the polar/longitudinal angle (bounds 0 to 2pi).

    dtheta_2 = (np.pi/3) / 2
    theta_c = dtheta_2
    dphi_2 = (np.pi/3) / 2
    phi_c = dphi_2
    left_edge = np.array([0.8, theta_c - dtheta_2, phi_c - dphi_2])
    right_edge = np.array([1.2, theta_c + dtheta_2, phi_c + dphi_2])

    assert scp._select_bbox(left_edge, right_edge)