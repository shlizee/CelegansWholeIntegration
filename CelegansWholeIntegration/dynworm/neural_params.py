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
Gg_Static_v3_5_extrapolate = np.load('Gg_v3_5_extrapolate.npy') # Cook et al (not weighted) + haspel

""" Synaptic connections (Chemical, 279*279) """
Gs_Static_v3_5_extrapolate = np.load('Gs_v3_5_extrapolate.npy') # Cook et al (not weighted) + haspel

""" Old versions """
Gg_Static_v1 = np.load('Gg_v1.npy') # Varshney et al
Gs_Static_v1 = np.load('Gs_v1.npy') # Varshney et al

Gg_Static = np.load('Gg_v3.npy') # Varshney et al + Haspel
Gs_Static = np.load('Gs_v3.npy') # Varshney et al + Haspel

Gg_Static_v3_5 = np.load('Gg_v3_5.npy') # Cook et al (not weighted)
Gs_Static_v3_5 = np.load('Gs_v3_5.npy') # Cook et al (not weighted)

""" Directionality (279*279) """
EMat_mask = np.load('emask_mat_v1.npy')

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
iext: input stimulus current amplitude scaler (use 10 for pA units)
nonlinear_AWA: whether the simulation will incorporate nonlinear channels for AWA
nonlinear_AVL: whether the simulation will incorporate nonlinear channels for AVL
"""

init_key_counts = 15

pA_unit_baseline = {

    "N" : 279, 
    "Gc" : 0.1,
    "C" : 0.015,
    "ggap" : 1.0,
    "gsyn" : 1.0,
    "Ec" : -35.0,
    "E_rev": -48.0, 
    "ar" : 1.0/1.5,
    "ad" : 5.0/1.5,
    "B" : 0.125,
    "rate" : 0.025,
    "offset" : 0.15,
    "iext" : 10.,
    "nonlinear_AWA": False,
    "nonlinear_AVL": False

    }

AWA_nonlinear_params = {

    "AWA_inds": np.array([73, 82]), 
    "gK": 1.5*10,
    "vK": -84.,
    "vk1": 2.,
    "vk2": 10.,
    "TK": .03,
    "gCa": 0.1*10,
    "vCa": 120,
    "fac": 0.4,
    "TC1": .001,
    "TC2": .08,
    "mx": 3.612/4,
    "vm1": -21.6,
    "vm2": 9.17,
    "vm3": 16.2,
    "vm4": -16.1,
    "vt1": 20.,
    "vt2": 24,
    "gK2": 0.8 * 10,
    "gKI": 5.,
    "gK4": 1.*10,
    "TS": 1.,
    "vs1": 13.,
    "vs2": 20.,
    "gK7": 0.1*10,
    "fc": 1.1,
    "fc2": 1.5,
    "TS2": 1.,
    "vq1": -25.,
    "vq2": 5.,
    "vb1": -42.,
    "vb2": 5.,
    "gK3": 0.3*10*1.1,
    "TbK": 1.2*1.5,
    "gK6": 0,
    "gK5": 0.7*10*1.1,
    "TKL": 18.*1.5,
    "TKH": 2*1.5,
    "vtk1": -52,
    "vtk2": 20.,
    "vp1": -32.,
    "vp2": 2.
}

AVL_nonlinear_params = {

    "AVL_ind": 124,
    "C_AVL": 0.05,
    "vK": -84,
    "vCA": 100,
    "gL": 2.7,
    "vL": -60,
    "g_NCA": 0.2,
    "vNA": 30,
    "g_UNC2": 10,
    "a_m_UNC2_a": 100,
    "a_m_UNC2_b": 25,
    "a_m_UNC2_c": 10,
    "b_m_UNC2_a": 400,
    "b_m_UNC2_b": -25,
    "b_m_UNC2_c":18,
    "a_h_UNC2_a": 10,
    "a_h_UNC2_b": -50,
    "a_h_UNC2_c": 10,
    "b_h_UNC2_a": 30,
    "b_h_UNC2_b": -17,
    "b_h_UNC2_c": 17,
    "g_EGL19": 7.5,
    "tau_m_EGL19_a": 0.0029,
    "tau_m_EGL19_b": -4.8,
    "tau_m_EGL19_c": 6,
    "tau_m_EGL19_d": 0.0019,
    "tau_m_EGL19_e": -8.6,
    "tau_m_EGL19_f": 30,
    "tau_m_EGL19_g": 0.0023,
    "tau_h_EGL19_a": 0.4,
    "tau_h_EGL19_b": 0.0446,
    "tau_h_EGL19_c": -33,
    "tau_h_EGL19_d": 5,
    "tau_h_EGL19_e": 0.0364,
    "tau_h_EGL19_f": 18.7,
    "tau_h_EGL19_g": 3.7,
    "tau_h_EGL19_h": 0.0431,
    "m_EGL19inf_v_eq": -4.4,
    "m_EGL19inf_k_a": 7.5,
    "h_EGL19inf_a": 1.43,
    "h_EGL19inf_v_eq_1": 14.9,
    "h_EGL19inf_k_i1": 12,
    "h_EGL19inf_b": 0.14,
    "h_EGL19inf_c": 5.96,
    "h_EGL19inf_v_eq_2": -20.5,
    "h_EGL19inf_k_i2": 8.1,
    "h_EGL19inf_d": 0.6,
    "g_CCA1": 2.5,
    "m_CCA1inf_v_eq": -43.32,
    "m_CCA1inf_k_a": 7.6,
    "tau_m_CCA1_a": 0.04,
    "tau_m_CCA1_b": -62.5,
    "tau_m_CCA1_c": -12.6,
    "tau_m_CCA1_d": 0.0007,
    "h_CCA1inf_v_eq": -58,
    "h_CCA1inf_k_i": 7,
    "tau_h_CCA1_a": 0.28,
    "tau_h_CCA1_b": -60.7,
    "tau_h_CCA1_c": 8.5,
    "tau_h_CCA1_d": 0.0198,
    "g_SHL1": 40,
    "m_SHL1inf_v_eq": -6.8,
    "m_SHL1inf_k_a": 14.1,
    "tau_m_SHL1_a": 0.0014,
    "tau_m_SHL1_b": -17.5,
    "tau_m_SHL1_c": 12.9,
    "tau_m_SHL1_d": -3.7,
    "tau_m_SHL1_e": 6.5,
    "tau_m_SHL1_f": 0.0002,
    "h_SHL1inf_v_eq": -33.1,
    "h_SHL1inf_k_i": 8.3,
    "tau_h_SHL1f_a": 0.0059,
    "tau_h_SHL1f_b": -8.2,
    "tau_h_SHL1f_c": 2.9,
    "tau_h_SHL1f_d": 0.00273,
    "tau_h_SHL1s_a": 0.0842,
    "tau_h_SHL1s_b": -7.7,
    "tau_h_SHL1s_c": 2.4,
    "tau_h_SHL1s_d": 0.0119,
    "g_EGL36": 13.5,
    "m_EGL36inf_v_eq": 63,
    "m_EGL36inf_k_a": 28.5,
    "tau_m_EGL36f_tau_f": 0.013,
    "tau_m_EGL36f_tau_m": 0.063,
    "tau_m_EGL36f_tau_s": 0.355,
    "g_EXP2": 31,
    "a_1_p_1": 24.1,
    "a_1_p_2": 0.0408,
    "b_1_p_3": 9.1,
    "b_1_p_4": 0.03,
    "K_f_p_5": 37.2,
    "K_b_p_6": 310,
    "a_2_p_7": 37.6,
    "a_2_p_8": 0.0472,
    "b_2_p_9": 1.5,
    "b_2_p_10": 0.0703,
    "a_i_p_11": 217.7,
    "a_i_p_12": 0.03,
    "b_i_p_13": 31.3,
    "b_i_p_14": 0.1418,
    "a_i2_p_15": 8.72e-3,
    "a_i2_p_16": 1.4e-6

    }