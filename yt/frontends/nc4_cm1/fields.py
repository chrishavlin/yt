"""
Skeleton-specific fields



"""

#-----------------------------------------------------------------------------
# Copyright (c) 2013, yt Development Team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
#-----------------------------------------------------------------------------

from yt.fields.field_info_container import \
    FieldInfoContainer

# We need to specify which fields we might have in our dataset.  The field info
# container subclass here will define which fields it knows about.  There are
# optionally methods on it that get called which can be subclassed.

'''
class StreamFieldInfo(FieldInfoContainer):
    known_other_fields = (
        ("density", ("code_mass/code_length**3", ["density"], None)),
        ("dark_matter_density", ("code_mass/code_length**3", ["dark_matter_density"], None)),
        ("number_density", ("1/code_length**3", ["number_density"], None)),
        ("pressure", ("dyne/code_length**2", ["pressure"], None)),
        ("thermal_energy", ("erg / g", ["thermal_energy"], None)),
        ("temperature", ("K", ["temperature"], None)),
        ("velocity_x", ("code_length/code_time", ["velocity_x"], None)),
        ("velocity_y", ("code_length/code_time", ["velocity_y"], None)),
        ("velocity_z", ("code_length/code_time", ["velocity_z"], None)),
        ("magnetic_field_x", ("gauss", [], None)),
        ("magnetic_field_y", ("gauss", [], None)),
        ("magnetic_field_z", ("gauss", [], None)),
        ("radiation_acceleration_x", ("code_length/code_time**2", ["radiation_acceleration_x"], None)),
        ("radiation_acceleration_y", ("code_length/code_time**2", ["radiation_acceleration_y"], None)),
        ("radiation_acceleration_z", ("code_length/code_time**2", ["radiation_acceleration_z"], None)),
        ("metallicity", ("Zsun", ["metallicity"], None)),

        # We need to have a bunch of species fields here, too
        ("metal_density",   ("code_mass/code_length**3", ["metal_density"], None)),
        ("hi_density",      ("code_mass/code_length**3", ["hi_density"], None)),
        ("hii_density",     ("code_mass/code_length**3", ["hii_density"], None)),
        ("h2i_density",     ("code_mass/code_length**3", ["h2i_density"], None)),
        ("h2ii_density",    ("code_mass/code_length**3", ["h2ii_density"], None)),
        ("h2m_density",     ("code_mass/code_length**3", ["h2m_density"], None)),
        ("hei_density",     ("code_mass/code_length**3", ["hei_density"], None)),
        ("heii_density",    ("code_mass/code_length**3", ["heii_density"], None)),
        ("heiii_density",   ("code_mass/code_length**3", ["heiii_density"], None)),
        ("hdi_density",     ("code_mass/code_length**3", ["hdi_density"], None)),
        ("di_density",      ("code_mass/code_length**3", ["di_density"], None)),
        ("dii_density",     ("code_mass/code_length**3", ["dii_density"], None)),
    )

'''

class CM1FieldInfo(FieldInfoContainer):
    known_other_fields = (
        # Each entry here is of the form
        # ( "name", ("units", ["fields", "to", "alias"], # "display_name")),
        ("uinterp", ("m/s",    ["velocity_x"], None)),
        ("vinterp", ("m/s",    ["velocity_y"], None)),
        ("winterp", ("m/s",    ["velocity_z"], None)),
        ("hwin_sr", ("m/s",    ["storm_relative_horizontal_wind_speed"], None)),
        ("windmag_sr", ("m/s", ["storm_relative_3D_wind_speed"], None)),
        ("hwin_gr", ("m/s",    ["ground_relative_horizontal_wind_speed"], None)),
        ("thpert", ("K",       ["potential_temperature_perturbation"], None)),
        ("thrhopert", ("K",    ["density_potential_temperature_perturbation"], None)),
        ("prespert", ("hPa",   ["presure_perturbation"], None)),
        ("rhopert", ("kg/m^3", ["density_perturbation"], None)),
        ("dbz", ("dBZ",        ["simulated_reflectivity"], None)),
        ("qvpert", ("g/kg",    ["water_vapor_mixing_ratio_perturbation"], None)),
        ("qc", ("g/kg",        ["cloud_liquid_water_mixing_ratio"], None)),
        ("qr", ("g/kg",        ["rain_mixing_ratio"], None)),
        ("qi", ("g/kg",        ["cloud_ice_mixing_ratio"], None)),
        ("qs", ("g/kg",        ["snow_mixing_ratio"], None)),
        ("qg", ("g/kg",        ["graupel_or_hail_mixing_ratio"], None)),
        ("qcloud", ("g/kg",    ["sum_of_cloud_water_and_cloud_ice_mixing_ratios"], None)),
        ("qprecip", ("g/kg",   ["sum_of_rain_graupel_snow_mixing_ratios"], None)),
        ("nci", ("cm^-3",      ["number_concerntration_of_cloud_ice"], None)),
        ("ncr", ("cm^-3",      ["number_concentration_of_rain"], None)),
        ("ncs", ("cm^-3",      ["number_concentration_of_snow"], None)),
        ("ncg", ("cm^-3",      ["number_concentration_of_graupel_or_hail"], None)),
        ("xvort", ("s^-1",     ["vorticity_x"], None)),
        ("yvort", ("s^-1",     ["vorticity_y"], None)),
        ("zvort", ("s^-1",     ["vorticity_z"], None)),
        ("hvort", ("s^-1",     ["horizontal_vorticity_magnitude"], None)),
        ("vortmag", ("s^-1",   ["vorticity_magnitude"], None)),
        ("streamvort", ("s^-1",["streamwise_vorticity"], None)),
        ("khh", ("m^2/s",      ["khh"], None)),
        ("khv", ("m^2/s",      ["khv"], None)),
        ("kmh", ("m^2/s",      ["kmh"], None)),
        ("kmv", ("m^2/s",      ["kmv"], None))
    )

    known_particle_fields = (
        # Identical form to above
        # ( "name", ("units", ["fields", "to", "alias"], # "display_name")),
    )

    def __init__(self, ds, field_list):
        super(CM1FieldInfo, self).__init__(ds, field_list)
        # If you want, you can check self.field_list

    def setup_fluid_fields(self):
        # Here we do anything that might need info about the dataset.
        # You can use self.alias, self.add_output_field (for on-disk fields)
        # and self.add_field (for derived fields).
        pass

    def setup_particle_fields(self, ptype):
        super(CM1FieldInfo, self).setup_particle_fields(ptype)
        # This will get called for every particle type.
