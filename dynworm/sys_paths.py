
# coding: utf-8

########################################################################################################################################################################
# PATHS FOR WINDOWS AND UNIX BASED OPERATING SYSTEMS ###################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

import os
import platform

platform = platform.system()
default_dir = os.getcwd()

if platform == 'Windows':

    main_dir = default_dir + '\dynworm'
    connectome_data_dir = main_dir + '\connectome_data'
    eigworm_data_dir = main_dir + '\eigenworm_modes'
    muscle_map_dir = main_dir + '\muscle_maps'
    inputmat_dir = main_dir + '\presets_input'
    voltagemat_dir = main_dir + '\presets_voltage'
    mldata_dir = main_dir + '\ml_data'
    vids_dir = default_dir + '\created_vids'

else:

    main_dir = default_dir + '/dynworm'
    connectome_data_dir = main_dir + '/connectome_data'
    eigworm_data_dir = main_dir + '/eigenworm_modes'
    muscle_map_dir = main_dir + '/muscle_maps'
    inputmat_dir = main_dir + '/presets_input'
    voltagemat_dir = main_dir + '/presets_voltage'
    mldata_dir = main_dir + '/ml_data'
    vids_dir = default_dir + '/created_vids'