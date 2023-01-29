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

""" Gap/Synaptic Conductance (279*279) """
ggap_total_mat = np.ones((279, 279)) * 0.1 # 100pS = 0.1nS
gsyn_max_mat = np.ones((279, 279)) * 0.1

""" Directionality (279*279) """
EMat_mask = np.load('emask_mat_v1.npy') # Describes which neurons are excitatory and which are inhibitory

# Other connectome variants

Gg_Static_v3_5_extrapolate = np.load('Gg_v3_5_extrapolate.npy') # Cook et al (not weighted) + haspel
Gs_Static_v3_5_extrapolate = np.load('Gs_v3_5_extrapolate.npy') # Cook et al (not weighted) + haspel

Gg_Static_v1 = np.load('Gg_v1.npy') # Varshney et al
Gs_Static_v1 = np.load('Gs_v1.npy') # Varshney et al

Gg_Static_v3_5 = np.load('Gg_v3_5.npy') # Cook et al (not weighted)
Gs_Static_v3_5 = np.load('Gs_v3_5.npy') # Cook et al (not weighted)

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

# COMPUTATION GRAPH

# 0. Initialize 279 nodes

# 1. Physiological parameters
# 2. Connectome/EI distribution
# 3. Resting potentials (Vth)
# 4. System initial conditions
# 5. Solve System Dynamics

# 6. System Dynamics -> Muscle inputs
# 7. Muscle inputs -> Calcium dynamics
# 8. Calcium dynamics -> forces
# 9. Solve visco-elastic model

init_key_counts = 15

pA_unit_baseline = {

    "N" : 279, 
    "Gc" : 0.01, # nS, 10pS
    "C" : 0.0015, # nF, 1.5pF
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

L_NL_t_conversion = 0.001 #1s to 1ms conversion

AWA_nonlinear_params = {

    "AWA_inds": np.array([73, 82]), #73, 82 
    "C": 0.0015, #0.0015,
    "gK": 1.5,                                           #0
    "vK": -84.,                                          #1
    "vk1": 2.,                                           #2
    "vk2": 10.,                                          #3
    "TK": 30 * L_NL_t_conversion,                        #4
    "gCa": 0.1, # 0.1 (wt)                               #5
    "vCa": 120,                                          #6
    "fac": 0.4,                                          #7
    "TC1": 1 * L_NL_t_conversion,                        #8
    "TC2": 80 * L_NL_t_conversion, #80 (wt)              #9
    "mx": 3.612/4,                                       #10
    "vm1": -21.6,                                        #11
    "vm2": 9.17,                                         #12
    "vm3": 16.2,                                         #13
    "vm4": -16.1,                                        #14
    "vt1": 20.,#20                                       #15
    "vt2": 24,                                           #16
    "gK2": 0.8, #0.8                                     #17
    "gKI": 5.,                                           #18
    "gK4": 1., # 1 (wt)                                  #19
    "TS": 1000. * L_NL_t_conversion,                     #20
    "vs1": 13.,                                          #21
    "vs2": 20.,                                          #22
    "gK7": 0.1,                                          #23
    "fc": 1.1,                                           #24
    "fc2": 1.5,                                          #25
    "TS2": 1000. * L_NL_t_conversion,                    #26
    "vq1": -25.,                                         #27
    "vq2": 5.,                                           #28
    "vb1": -42.,                                         #29
    "vb2": 5.,                                           #30
    "gK3": 0.3*1.1, #0.3*1.1 (wt)                        #31
    "TbK": 1200*1.5 * L_NL_t_conversion,                 #32
    "gK6": 0,                                            #33
    "gK5": 0.7*1.1, #0.7*1.1 (wt)                        #34
    "TKL": 18000*1.5 * L_NL_t_conversion,                #35
    "TKH": 2000*1.5 * L_NL_t_conversion,                 #36
    "vtk1": -52,                                         #37
    "vtk2": 20.,                                         #38
    "vp1": -32.,                                         #39
    "vp2": 2.,                                           #40
    "gL": 0.25,                                          #41
    "vL": -65                                            #42
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

########################################################################################################################################################################
# NEURAL PARAMETERS FOR CONDUCTANCE BASED MODELS #######################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

Cap_model_params = {
    
    "gCa": 0.68,            #0
    "gKir": 0.254,          #1
    "gK": 1.16,             #2
    "gL": 0.0002,           #3
    "ECa": 20.16,           #4
    "EK": -62.18,           #5  
    "EL": -37.6,            #6
    "v_eq_mCa": -5.5,       #7
    "v_eq_Kir": -65.7,      #8
    "v_eq_mK": -9.38,       #9  
    "v_eq_hK": -24.27,      #10
    "k_mCa": 1.6,           #11
    "k_Kir": -1.32,         #12
    "k_mK": 1.28,           #13
    "k_hK": -23.44,         #14
    "tau_mCa": 0.399,       #15
    "tau_mK": 0.03,         #16
    "tau_hK": 0.61,         #17
    "mCa_0": 0.002,         #18
    "mK_0": 0.643,          #19
    "hK_0": 0.113,          #20
    "C": 0.042,             #21
    "V_0": -38,             #22
    "iext": 0               #23
}

Cat_model_params = {
    
    "gCa": 0.124,           #0
    "gKir": 0.157,          #1
    "gK": 0.223,            #2
    "gL": 0.14,             #3
    "ECa": 135.9,           #4
    "EK": -98.23,           #5
    "EL": -41.07,           #6
    "v_eq_mCa": -19.09,     #7
    "v_eq_hCa": -21.24,     #8
    "v_eq_Kir": -90,        #9
    "v_eq_mK": -17.71,      #10
    "k_mCa": 4.67,          #11
    "k_hCa": -17.62,        #12
    "k_Kir": -30,           #13
    "k_mK": 7.39,           #14
    "tau_mCa": 0.0001,      #15
    "tau_hCa": 10.59,       #16
    "tau_mK": 0.0005,       #17
    "mCa_0": 0.001,         #18
    "hCa_0": 0.8,           #19
    "mK_0": 0.999,          #20
    "C": 0.04,              #21
    "V_0": -53,             #22
    "iext": 0               #23
}

########################################################################################################################################################################
# NEURAL PARAMETERS FOR GENERIC CBM ####################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

generic_model_params = {
    
    "SHL1_g": 2.9,                                            #0 (0, 50) nS
    "SHL1_m_inf_Veq": 11.2,                                   #1 (-90, 0) mV
    "SHL1_m_inf_ka": 14.1,                                    #2 (0, 30) mV
    "SHL1_h_inf_Veq": -33.1,                                  #3 (-90, 0) mV
    "SHL1_h_inf_ki": 8.3,                                     #4 (-30, 0) mV
    "SHL1_tau_m_a": 13.8,                                     #5 (0, 1500) ms
    "SHL1_tau_m_b": -17.5,                                    #6 (-90, 0) mV
    "SHL1_tau_m_c": 12.9,                                     #7 (0, 30) mV
    "SHL1_tau_m_d": -3.7,                                     #8 (-90, 0) mV
    "SHL1_tau_m_e": 6.5,                                      #9 (0, 30) mV
    "SHL1_tau_m_f": 1.9,                                      #10 (0, 1500) ms
    "SHL1_tau_h_f_a": 539.2,                                  #11 (0, 1500) ms
    "SHL1_tau_h_f_b": -28.2,                                  #12 (-90, 0) mV
    "SHL1_tau_h_f_c": 4.9,                                    #13 (-30, 0) mV
    "SHL1_tau_h_f_d": 27.3,                                   #14 (0, 1500) ms
    "SHL1_tau_h_s_a": 8422.,                                  #15 (0, 1500) ms
    "SHL1_tau_h_s_b": -37.7,                                  #16 (-90, 0) mV
    "SHL1_tau_h_s_c": 6.4,                                    #17 (-30, 0) mV
    "SHL1_tau_h_s_d": 118.9,                                  #18 (0, 1500) ms
    "SHK1_g": 0.1,                                            #19 (0, 50) nS
    "SHK1_m_inf_Veq": 20.4,                                   #20 (-90, 0) mV
    "SHK1_m_inf_ka": 7.7,                                     #21 (0, 30) mV
    "SHK1_h_inf_Veq": -7.,                                    #22 (-90, 0) mV
    "SHK1_h_inf_ki": 5.8,                                     #23 (-30, 0) mV
    "SHK1_tau_m_a": 26.6,                                     #24 (0, 1500) ms
    "SHK1_tau_m_b": -33.7,                                    #25 (-90, 0) mV
    "SHK1_tau_m_c": 15.8,                                     #26 (0, 30) mV
    "SHK1_tau_m_d": -33.7,                                    #27 (-90, 0) mV
    "SHK1_tau_m_e": 15.4,                                     #28 (0, 30) mV
    "SHK1_tau_m_f": 2.,                                       #29 (0, 1500) ms
    "SHK1_tau_h_a": 1400.,                                    #30 (0, 1500) ms
    "KVS1_g": 0.8,                                            #31 (0, 50) nS
    "KVS1_m_inf_Veq": 57.1,                                   #32 (-90, 0) mV
    "KVS1_m_inf_ka": 25.,                                     #33 (0, 30) mV
    "KVS1_h_inf_Veq": 47.3,                                   #34 (-90, 0) mV
    "KVS1_h_inf_ki": 11.1,                                    #35 (-30, 0) mV
    "KVS1_tau_m_a": 30.,                                      #36 (0, 1500) ms
    "KVS1_tau_m_b": 18.1,                                     #37 (-90, 0) mV
    "KVS1_tau_m_c": 20.,                                      #38 (0, 30) mV
    "KVS1_tau_m_d": 1.,                                       #39 (0, 1500) ms
    "KVS1_tau_h_a": 88.5,                                     #40 (0, 1500) ms
    "KVS1_tau_h_b": 50.,                                      #41 (-90, 0) mV
    "KVS1_tau_h_c": 15.,                                      #42 (-30, 0) mV
    "KVS1_tau_h_d": 53.4,                                     #43 (0, 1500) ms
    "KQT3_g": 0.55,                                           #44 (0, 50) nS
    "KQT3_m_inf_Veq": -12.8,                                  #45 (-90, 0) mV
    "KQT3_m_inf_ka": 15.8,                                    #46 (0, 30) mV
    "KQT3_w_inf_Veq": -1.1,                                   #47 (-90, 0) mV
    "KQT3_w_inf_ki": 28.8,                                    #48 (-30, 0) mV
    "KQT3_w_inf_a": 0.5,                                      #49 NT 1
    "KQT3_w_inf_b": 0.5,                                      #50 NT 1
    "KQT3_s_inf_Veq": -45.3,                                  #51 (-90, 0) mV
    "KQT3_s_inf_ki": 12.3,                                    #52 (-30, 0) mV
    "KQT3_s_inf_a": 0.3,                                      #53 NT 1
    "KQT3_s_inf_b": 0.7,                                      #54 NT 1
    "KQT3_tau_m_f_a": 395.3,                                  #55 (0, 1500) ms
    "KQT3_tau_m_f_b": 38.1,                                   #56 (-90, 0) mV
    "KQT3_tau_m_f_c": 33.6,                                   #57 (0, 30) mV
    "KQT3_tau_m_s_a": 5503.,                                  #58 (0, 1500) ms
    "KQT3_tau_m_s_b": -5345.4,                                #59 (0, 1500) ms
    "KQT3_tau_m_s_c": 0.0283,                                 #60 NT 0.01
    "KQT3_tau_m_s_d": -23.9,                                  #61 (-90, 0) mV
    "KQT3_tau_m_s_e": -4590.,                                 #62 (0, 1500) ms
    "KQT3_tau_m_s_f": 0.0357,                                 #63 NT 0.01
    "KQT3_tau_m_s_g": 14.2,                                   #64 (-90, 0) mV
    "KQT3_tau_w_a": 0.5,                                      #65 (0, 1500) ms
    "KQT3_tau_w_b": 2.9,                                      #66 (0, 1500) ms
    "KQT3_tau_w_c": -48.1,                                    #67 (-90, 0) mV
    "KQT3_tau_w_d": 48.8,                                     #68 (-30, 0) mV
    "KQT3_tau_s_a": 500.,                                     #69 (0, 1500) ms
    "EGL2_g": 0.85,                                           #70 (0, 50) nS
    "EGL2_m_inf_Veq": -6.9,                                   #71 (-90, 0) mV
    "EGL2_m_inf_ka": 14.9,                                    #72 (0, 30) mV
    "EGL2_tau_m_a": 1845.8,                                   #73 (0, 1500) ms
    "EGL2_tau_m_b": -122.6,                                   #74 (-90, 0) mV
    "EGL2_tau_m_c": 13.8,                                     #75 (0, 30) mV
    "EGL2_tau_m_d": 1517.74,                                  #76 (0, 1500) ms
    "EGL36_g": 0.,                                            #77 (0, 50) nS
    "EGL36_m_inf_Veq": 63.,                                   #78 (-90, 0) mV
    "EGL36_m_inf_ka": 28.5,                                   #79 (0, 30) mV
    "EGL36_tau_m_s_a": 355.,                                  #80 (0, 1500) ms
    "EGL36_tau_m_m_a": 63.,                                   #81 (0, 1500) ms
    "EGL36_tau_m_f_a": 13.,                                   #82 (0, 1500) ms
    "IRK_g": 0.25,                                            #83 (0, 50) nS
    "IRK_m_inf_Veq": -82.,                                    #84 (-90, 0) mV
    "IRK_m_inf_ka": 13.,                                      #85 (0, 30) mV
    "IRK_tau_m_a": 17.1,                                      #86 (0, 1500) ms
    "IRK_tau_m_b": -17.8,                                     #87 (-90, 0) mV
    "IRK_tau_m_c": 20.3,                                      #88 (0, 30) mV
    "IRK_tau_m_d": -43.4,                                     #89 (-90, 0) mV
    "IRK_tau_m_e": 11.2,                                      #90 (0, 30) mV
    "IRK_tau_m_f": 3.8,                                       #91 (0, 1500) ms
    "EGL19_g": 1.55,                                          #92 (0, 50) nS
    "EGL19_m_inf_Veq": 5.6,                                   #93 (-90, 0) mV
    "EGL19_m_inf_ka": 7.5,                                    #94 (0, 30) mV
    "EGL19_h_inf_Veq": 24.9,                                  #95 (-90, 0) mV
    "EGL19_h_inf_ki": 12.,                                    #96 (-30, 0) mV
    "EGL19_h_inf_Veq_b": -10.5,                               #97 (-90, 0) mV
    "EGL19_h_inf_ki_b": 8.1,                                  #98 (-30, 0) mV
    "EGL19_h_inf_a": 1.43,                                    #99 NT 1
    "EGL19_h_inf_b": 0.14,                                    #100 NT 1
    "EGL19_h_inf_c": 5.96,                                    #101 NT 1
    "EGL19_h_inf_d": 0.6,                                     #102 NT 1
    "EGL19_tau_m_a": 2.9,                                     #103 (0, 1500) ms
    "EGL19_tau_m_b": 5.2,                                     #104 (-90, 0) mV
    "EGL19_tau_m_c": 6.,                                      #105 (0, 30) mV
    "EGL19_tau_m_d": 1.9,                                     #106 (0, 1500) ms
    "EGL19_tau_m_e": 1.4,                                     #107 (-90, 0) mV
    "EGL19_tau_m_f": 30.,                                     #108 (0, 30) mV
    "EGL19_tau_m_g": 2.3,                                     #109 (0, 1500) ms
    "EGL19_tau_h_a": 0.4,                                     #110 NT 1
    "EGL19_tau_h_b": 44.6,                                    #111 (0, 1500) ms
    "EGL19_tau_h_c": -23.,                                    #112 (-90, 0) mV
    "EGL19_tau_h_d": 5.,                                      #113 (-30, 0) mV
    "EGL19_tau_h_e": 36.4,                                    #114 (0, 1500) ms
    "EGL19_tau_h_f": 28.7,                                    #115 (-90, 0) mV
    "EGL19_tau_h_g": 3.7,                                     #116 (-30, 0) mV
    "EGL19_tau_h_h": 43.1,                                    #117 (0, 1500) ms
    "UNC2_g": 1.,                                             #118 (0, 50) nS 
    "UNC2_m_inf_Veq": -12.2,                                  #119 (-90, 0) mV
    "UNC2_m_inf_ka": 4.,                                      #120 (0, 30) mV
    "UNC2_h_inf_Veq": -52.5,                                  #121 (-90, 0) mV
    "UNC2_h_inf_ki": 5.6,                                     #122 (-30, 0) mV
    "UNC2_tau_m_a": 1.5,                                      #123 (0, 1500) ms
    "UNC2_tau_m_b": -8.2,                                     #124 (-90, 0) mV
    "UNC2_tau_m_c": 9.1,                                      #125 (0, 30) mV
    "UNC2_tau_m_d": 15.4,                                     #126 (0, 30) mV
    "UNC2_tau_m_e": 0.1,                                      #127 (0, 1500) ms
    "UNC2_tau_h_a": 83.8,                                     #128 (0, 1500) ms
    "UNC2_tau_h_b": 52.9,                                     #129 (-90, 0) mV
    "UNC2_tau_h_c": -3.5,                                     #130 (-30, 0) mV
    "UNC2_tau_h_d": 72.1,                                     #131 (0, 1500) ms
    "UNC2_tau_h_e": 23.9,                                     #132 (-90, 0) mV
    "UNC2_tau_h_f": -3.6,                                     #133 (-30, 0) mV
    "CCA1_g": 0.7,                                            #134 (0, 50) nS
    "CCA1_m_inf_Veq": -43.32,                                 #135 (-90, 0) mV
    "CCA1_m_inf_ka": 7.6,                                     #136 (0, 30) mV
    "CCA1_h_inf_Veq": -58.,                                   #137 (-90, 0) mV
    "CCA1_h_inf_ki": 7.,                                      #138 (-30, 0) mV
    "CCA1_tau_m_a": 40.,                                      #139 (0, 1500) ms
    "CCA1_tau_m_b": -62.5,                                    #140 (-90, 0) mV
    "CCA1_tau_m_c": -12.6,                                    #141 (0, 30) mV
    "CCA1_tau_m_d": 0.7,                                      #142 (0, 1500) ms
    "CCA1_tau_h_a": 280.,                                     #143 (0, 1500) ms
    "CCA1_tau_h_b": -60.7,                                    #144 (-90, 0) mV
    "CCA1_tau_h_c": 8.5,                                      #145 (-30, 0) mV
    "CCA1_tau_h_d": 19.8,                                     #146 (0, 1500) ms
    "SLO1_g": 0.11,                                           #147 (0, 50) nS
    "SLO1_wyx": 0.013,                                        #148 NT 0.01
    "SLO1_wxy": -0.028,                                       #149 NT 0.01
    "SLO1_w0_neg": 3.15,                                      #150 NT 0.001
    "SLO1_w0_pos": 0.16,                                      #151 NT 0.001
    "SLO1_Kxy": 55.73,                                        #152 NT 1
    "SLO1_nxy": 1.3,                                          #153 NT 1
    "SLO1_Kyx": 34.34,                                        #154 NT 1
    "SLO1_nyx": 1e-4,                                         #155 NT 1
    "SLO2_g": 0.1,                                            #156 (0, 50) nS
    "SLO2_wyx": 0.019,                                        #157 NT 0.01
    "SLO2_wxy": -0.024,                                       #158 NT 0.01
    "SLO2_w0_neg": 0.9,                                       #159 NT 0.001
    "SLO2_w0_pos": 0.027,                                     #160 NT 0.001
    "SLO2_Kxy": 93.45,                                        #161 NT 1
    "SLO2_nxy": 1.84,                                         #162 NT 1
    "SLO2_Kyx": 3294.55,                                      #163 NT 1
    "SLO2_nyx": 1e-5,                                         #164 NT 1
    "KCNL_g": 0.06,                                           #165 (0, 50) nS
    "KCNL_KCa": 0.33,                                         #166 NT 1
    "KCNL_tau_m_a": 6.3,                                      #167 (0, 1500) ms
    "ICC_gsc": 0.04,                                          #168 (0, 50) nS
    "ICC_r": 13. * 1e-3,                                      #169 NT 1
    "ICC_F": 96485. * 1e-15,                                  #170 NT 1
    "ICC_DCa": 250.,                                          #171 NT 1
    "ICC_KB_pos": 500.,                                       #172 NT 1
    "ICC_B_tot": 30.,                                         #173 NT 1
    "ICC_Ca2_pos_n_ci": 0.05,                                 #174 NT 1
    "ICC_Vcell": 31.16,                                       #175 NT 1
    "ICC_f": 0.001,                                           #176 NT 1
    "ICC_tau_Ca": 50.,                                        #177 NT 1000
    "ICC_Ca2_pos_m_eq": 0.05,                                 #178 NT 1
    "NCA_g": 0.06,                                            #179 (0, 50) nS
    "LEAK_g": 0.27,                                           #180 (0, 50) nS
    "VK": -80.,                                               #181 (-100, 0) mV
    "VCa": 60.,                                               #182 (20, 150) mV
    "VNa": 30.,                                               #183 (20, 150) mV
    "VL": -90.,                                               #184 (-80, 30) mV
    "V0": -70.54,                                             #185 Random (-90, 0) mV
    "m_SHL1_0": 0.,                                           #186 (0, 1)
    "h_SHL1_f_0": 0.,                                         #187 (0, 1)
    "h_SHL1_s_0": 0.,                                         #188 (0, 1)
    "m_KVS1_0": 0.,                                           #189 (0, 1)
    "h_KVS1_0": 0.,                                           #190 (0, 1)
    "m_SHK1_0": 0.,                                           #191 (0, 1)
    "h_SHK1_0": 0.,                                           #192 (0, 1)
    "m_KQT3_f_0": 0.,                                         #193 (0, 1)
    "m_KQT3_s_0": 0.,                                         #194 (0, 1)
    "w_KQT3_0": 0.,                                           #195 (0, 1)
    "s_KQT3_0": 0.,                                           #196 (0, 1)
    "m_EGL2_0": 0.,                                           #197 (0, 1)
    "m_EGL36_f_0": 0.,                                        #198 (0, 1)
    "m_EGL36_m_0": 0.,                                        #199 (0, 1)
    "m_EGL36_s_0": 0.,                                        #200 (0, 1)
    "m_IRK_0": 0.,                                            #201 (0, 1)
    "m_EGL19_0": 0.,                                          #202 (0, 1)
    "h_EGL19_0": 0.,                                          #203 (0, 1)
    "m_UNC2_0": 0.,                                           #204 (0, 1)
    "h_UNC2_0": 0.,                                           #205 (0, 1) 
    "m_CCA1_0": 0.,                                           #206 (0, 1)
    "h_CCA1_0": 0.,                                           #207 (0, 1)
    "m_SLO1_EGL19_0": 0.,                                     #208 (0, 1)
    "m_SLO1_UNC2_0": 0.,                                      #209 (0, 1)
    "m_SLO2_EGL19_0": 0.,                                     #210 (0, 1)
    "m_SLO2_UNC2_0": 0.,                                      #211 (0, 1)
    "m_KCNL_0": 0.,                                           #212 (0, 1)
    "CA2_pos_m_0": 0.05,                                      #213 NT 1
    "C": 0.0031,                                              #214 (0, 10) nF
    "Iext": 0.001                                             #215 NT 1 pA

}