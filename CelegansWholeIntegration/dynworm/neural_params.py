# coding: utf-8

import os
import numpy as np

from dynworm import sys_paths as paths

########################################################################################################################################################################
# PARAMETERS / CONFIGURATION ###########################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

os.chdir(paths.connectome_data_dir)

""" Gap junctions (Chemical, 279*279) """ 
Gg_Static = np.load('Gg_v3.npy') # Varshney et al + Haspel. The is the gap connectome data used by the model.

""" Synaptic connections (Chemical, 279*279) """
Gs_Static = np.load('Gs_v3.npy') # Varshney et al + Haspel. The is the synaptic connectome data used by the model.

ggap_total_mat = np.ones((279, 279)) * 0.1 # 100pS = 0.1nS
gsyn_max_mat = np.ones((279, 279)) * 0.1

# Other connectome variants

Gg_Static_v3_5_extrapolate = np.load('Gg_v3_5_extrapolate.npy') # Cook et al (not weighted) + haspel
Gs_Static_v3_5_extrapolate = np.load('Gs_v3_5_extrapolate.npy') # Cook et al (not weighted) + haspel

Gg_Static_v1 = np.load('Gg_v1.npy') # Varshney et al
Gs_Static_v1 = np.load('Gs_v1.npy') # Varshney et al

Gg_Static_v3_5 = np.load('Gg_v3_5.npy') # Cook et al (not weighted)
Gs_Static_v3_5 = np.load('Gs_v3_5.npy') # Cook et al (not weighted)

""" Directionality (279*279) """
EMat_mask = np.load('emask_mat_v1.npy') # Describes which neurons are excitatory and which are inhibitory

os.chdir(paths.default_dir)

########################################################################################################################################################################
# NEURAL PARAMETERS FOR NETWORK SIMULATIONS ############################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

"""
N: Number of Neurons (279)
Gc: Cell membrane conductance (10pS)
C: Cell Membrane Capacitance (1.5pF)
ggap: Gap Junctions scaler (100pS Electrical, 279*279)
gsyn: Synaptic connections scaler (100pS Chemical, 279*279)
Ec: Leakage potential (-35mV) 
Ej: Reversal potential (-48mV)
ar: Synaptic activity's rise time (1/1.5) 
ad: Synaptic activity's decay time (5/1.5)
B: Width of the sigmoid (0.125mv^-1)
rate: Rate for continuous stimuli transition
offset: Offset for continuous stimuli transition
iext: input stimulus current amplitude scaler (use 1 for pA units)
nonlinear_AWA: whether the simulation will incorporate nonlinear channels for AWA
nonlinear_AVL: whether the simulation will incorporate nonlinear channels for AVL
"""

#TODO: IF SIMULAIONS TOO SLOW -> TWO DIFFERENT dt WHEN LINEAR ONLY VS NON_LINEAR NEURONS

init_key_counts = 14

pA_unit_baseline = {

    "N" : 279, 
    "Gc" : 0.01, # nS
    "C" : 0.0015, # nF
    "Ec" : -35.0, # mV
    "E_rev": -48.0, # mV
    "ar" : 1.0/1.5, 
    "ad" : 5.0/1.5,
    "B" : 0.125,
    "rate" : 0.025,
    "offset" : 0.15,
    "iext" : 1, # scaler to pA
    "dt": 0.01, # 1 -> 1s
    "time_scaler": 1, #conversion factor of 1 dt to second
    "nonlinear_AWA": False,
    "nonlinear_AVL": False

    }

L_NL_t_conversion = 0.001

AWA_nonlinear_params = {

    "AWA_inds": np.array([73, 82]), #73, 82 
    "C": 0.0015 * 0.01,#0.0015,
    "gK": 1.5,
    "vK": -84.,
    "vk1": 2.,
    "vk2": 10.,
    "TK": 30 * L_NL_t_conversion,
    "gCa": 0.1,
    "vCa": 120,
    "fac": 0.4,
    "TC1": 1 * L_NL_t_conversion,
    "TC2": 80 * L_NL_t_conversion,
    "mx": 3.612/4,
    "vm1": -21.6,
    "vm2": 9.17,
    "vm3": 16.2,
    "vm4": -16.1,
    "vt1": 20.,
    "vt2": 24,
    "gK2": 0.8,
    "gKI": 5.,
    "gK4": 1.,
    "TS": 1000. * L_NL_t_conversion,
    "vs1": 13.,
    "vs2": 20.,
    "gK7": 0.1,
    "fc": 1.1,
    "fc2": 1.5,
    "TS2": 1000. * L_NL_t_conversion,
    "vq1": -25.,
    "vq2": 5.,
    "vb1": -42.,
    "vb2": 5.,
    "gK3": 0.3*1.1,
    "TbK": 1200*1.5 * L_NL_t_conversion,
    "gK6": 0,
    "gK5": 0.7*1.1,
    "TKL": 18000*1.5 * L_NL_t_conversion,
    "TKH": 2000*1.5 * L_NL_t_conversion,
    "vtk1": -52,
    "vtk2": 20.,
    "vp1": -32.,
    "vp2": 2.,
    "gL": 0.25,
    "vL": -65
}

AVL_nonlinear_params = {

    "AVL_ind": np.array([124]),
    "C": 0.005*0.01,
    "g_L": 0.27,
    "v_Ca": 100,
    "v_K": -84,
    "v_L": -60,
    "g_NCA": 0.02,
    "v_Na": 30,
    "g_UNC2": 1,
    "m_a": 0.1* (1/L_NL_t_conversion),
    "m_b": 25,
    "m_c": 10,
    "m_d": 0.4* (1/L_NL_t_conversion),
    "m_e": -25,
    "m_f": 18,
    "h_a": 0.01* (1/L_NL_t_conversion),
    "h_b": -50,
    "h_c": 10,
    "h_d": 0.03* (1/L_NL_t_conversion),
    "h_e": -17,
    "h_f": 17,
    "g_EGL19": 0.75,
    "s_1": 2.9* L_NL_t_conversion,
    "s_2": -4.8,
    "s_3": 6,
    "s_4": 1.9* L_NL_t_conversion,
    "s_5": -8.6,
    "s_6": 30,
    "s_7": 2.3* L_NL_t_conversion,
    "s_8": 0.4,
    "s_9": 44.6* L_NL_t_conversion,
    "s_10": -33,
    "s_11": 5,
    "s_12": 36.4* L_NL_t_conversion,
    "s_13": 18.7,
    "s_14": 3.7,
    "s_15": 43.1* L_NL_t_conversion,
    "q_1": -4.4,
    "q_2": 7.5,
    "q_3": 1.43,
    "q_4": 14.9,
    "q_5": 12,
    "q_6": 0.14,
    "q_7": 5.96,
    "q_8": -20.5,
    "q_9": 8.1,
    "q_10": 0.6,
    "g_CCA1": 0.25,
    "c_1": -43.32,
    "c_2": 7.6,
    "c_3": 40* L_NL_t_conversion,
    "c_4": -62.5,
    "c_5": -12.6,
    "c_6": 0.7* L_NL_t_conversion,
    "d_1": -58,
    "d_2": 7,
    "d_3": 280* L_NL_t_conversion,
    "d_4": -60.7,
    "d_5": 8.5,
    "d_6": 19.8* L_NL_t_conversion,
    "g_SHL1": 4,
    "v_1": -6.8,
    "v_2": 14.1,
    "a_m": 1.4* L_NL_t_conversion,
    "b_m": -17.5,
    "c_m": 12.9,
    "d_m": -3.7,
    "e_m": 6.5,
    "f_m": 0.2* L_NL_t_conversion,
    "v_3": -33.1,
    "v_4": 8.3,
    "a_hf": 5.9* L_NL_t_conversion,
    "b_hf": -8.2,
    "c_hf": 2.9,
    "d_hf": 2.73* L_NL_t_conversion,
    "a_hs": 84.2* L_NL_t_conversion,
    "b_hs": -7.7,
    "c_hs": 2.4,
    "d_hs": 11.9* L_NL_t_conversion,
    "g_EGL36": 1.35,
    "e_1": 63,
    "e_2": 28.5,
    "t_f": 13* L_NL_t_conversion,
    "t_m": 63* L_NL_t_conversion,
    "t_s": 355* L_NL_t_conversion,
    "g_EXP2": 3.1,
    "p_1": 0.0241* (1/L_NL_t_conversion),
    "p_2": 0.0408,
    "p_3": 0.0091* (1/L_NL_t_conversion),
    "p_4": 0.03,
    "p_5": 0.0372* (1/L_NL_t_conversion),
    "p_6": 0.31* (1/L_NL_t_conversion),
    "p_7": 0.0376* (1/L_NL_t_conversion),
    "p_8": 0.0472,
    "p_9": 0.0015* (1/L_NL_t_conversion),
    "p_10": 0.0703,
    "p_11": 0.2177* (1/L_NL_t_conversion),
    "p_12": 0.03,
    "p_13": 0.0313* (1/L_NL_t_conversion),
    "p_14": 0.1418,
    "p_15": 8.72e-6* (1/L_NL_t_conversion),
    "p_16": 1.4011e-6

    }