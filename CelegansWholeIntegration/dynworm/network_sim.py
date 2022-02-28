
# coding: utf-8

########################################################################################################################################################################
# NETWORK SIMULATION MODULE ############################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

import time
import os

import numpy as np
import scipy.io as sio
from scipy import integrate, sparse, linalg, interpolate

from dynworm import sys_paths as paths
from dynworm import neural_params as n_params
from dynworm import body_params as b_params
from dynworm import utils

np.random.seed(10)

########################################################################################################################################################################
########################################################################################################################################################################
### BASE ENVIRONMENT INITIALIZATION ####################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

def initialize_params_neural(custom_baseline_params = False, additional_params = False):

    global params_obj_neural

    if custom_baseline_params == False:

        params_obj_neural = n_params.pA_unit_baseline.copy()
        print('Using the default neural parameters')

        if additional_params != False:

            params_obj_neural['nonlinear_AWA'] = additional_params['AWA']
            params_obj_neural['nonlinear_AVL'] = additional_params['AVL']
            print('Using additional neural parameters')

    else:

        assert type(custom_baseline_params) == dict, "Custom neural parameters should be of dictionary format"

        if validate_custom_neural_params(custom_baseline_params) == True:

            params_obj_neural = custom_baseline_params.copy()
            print('Accepted the custom neural parameters')

            if additional_params != False:

                params_obj_neural['nonlinear_AWA'] = additional_params['AWA']
                params_obj_neural['nonlinear_AVL'] = additional_params['AVL']
                print('Using additional neural parameters')

def validate_custom_neural_params(custom_baseline_params):

    # TODO: Also check for dimensions

    key_checker = []

    for key in n_params.default_baseline.keys():
        
        key_checker.append(key in custom_baseline_params)

    all_keys_present = np.sum(key_checker) == n_params.init_key_counts
    
    assert np.sum(key_checker) == n_params.init_key_counts, "Provided dictionary is incomplete"

    return all_keys_present

def initialize_connectivity(custom_connectivity_dict = False):

    # To be executed after load_params_neural
    # custom_connectivity_dict should be of dict format with keys - 'gap', 'syn', 'directionality'
    # TODO: Check validity of custom connectomes

    assert 'params_obj_neural' in globals(), "Neural parameters must be initialized before initializing the connectivity"

    if custom_connectivity_dict == False:

        params_obj_neural['Gg_Static'] = n_params.Gg_Static.copy()
        params_obj_neural['Gs_Static'] = n_params.Gs_Static.copy()
        EMat_mask = n_params.EMat_mask.copy()
        print('Using the default connectivity')

    else:

        assert type(custom_connectivity_dict) == dict, "Custom connectivity should be of dictionary format"

        params_obj_neural['Gg_Static'] = custom_connectivity_dict['gap'].copy()
        params_obj_neural['Gs_Static'] = custom_connectivity_dict['syn'].copy()
        EMat_mask = custom_connectivity_dict['directionality'].copy()
        print('Accepted the custom connectivity')

    params_obj_neural['EMat'] = params_obj_neural['E_rev'] * EMat_mask
    params_obj_neural['mask_Healthy'] = np.ones(params_obj_neural['N'], dtype = 'bool')


########################################################################################################################################################################
### MASTER EXECUTION FUNCTIONS (BASIC) #################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

def run_network_constinput(t_duration, input_vec, ablation_mask, \
    t_delta = 0.01, custom_initcond = False, ablation_type = "all", verbose = True):

    assert 'params_obj_neural' in globals(), "Neural parameters and connectivity must be initialized before running the simulation"

    """ Set up simulation parameters """

    tf = t_duration
    dt = t_delta # recommend 0.01

    nsteps = int(np.floor(tf/dt) + 1)
    params_obj_neural['inmask'] = input_vec
    progress_milestones = np.linspace(0, nsteps, 10).astype('int')

    """ define the connectivity """

    modify_Connectome(ablation_mask, ablation_type)

    """ Calculate V_threshold """

    params_obj_neural['vth'] = EffVth_rhs(params_obj_neural['inmask'])

    """ Set initial condition """

    if type(custom_initcond) == bool:

        initcond = compute_initcond_neural()

    else:

        initcond = custom_initcond
        print("using the custom initial condition")

    """ Configuring the ODE Solver """

    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

        r = integrate.ode(membrane_voltageRHS_constinput, compute_jacobian_constinput).set_integrator('vode', atol = 1e-3, method = 'bdf')

    else:

        r = integrate.ode(membrane_voltageRHS_constinput).set_integrator('vode', atol = 1e-3, method = 'bdf', with_jacobian = True)
     
    r.set_initial_value(initcond, 0)

    """ Additional Python step to store the trajectories """
    t = np.zeros(nsteps)
    traj = np.zeros((nsteps, params_obj_neural['N']))

    t[0] = 0
    traj[0, :] = initcond[:params_obj_neural['N']]
    vthmat = np.tile(params_obj_neural['vth'], (nsteps, 1))

    print("Network integration prep completed...")

    """ Integrate the ODE(s) across each delta_t timestep """
    print("Computing network dynamics...")
    k = 1

    while r.successful() and k < nsteps:

        r.integrate(r.t + dt)

        t[k] = r.t
        traj[k, :] = r.y[:params_obj_neural['N']]

        k += 1

        if verbose == True:

            if k in progress_milestones:

                print(str(np.round((float(k) / nsteps) * 100, 1)) + '% ' + 'completed')

    result_dict_network = {
            "t": t,
            "dt": dt,
            "steps": nsteps,
            "raw_v_solution": traj,
            "v_threshold": vthmat,
            "v_solution" : voltage_filter(np.subtract(traj, vthmat), 200, 1)
            }

    return result_dict_network

def run_network_dyninput(input_mat, ablation_mask, \
    t_delta = 0.01, custom_initcond = False, ablation_type = "all", \
    interp_kind_input = 'linear', verbose = True):

    assert 'params_obj_neural' in globals(), "Neural parameters and connectivity must be initialized before running the simulation"

    """ define pre-computed noise input matrix """

    t0 = 0
    tf = (len(input_mat) - 1) * t_delta # the last timepoint to be computed
    dt = t_delta

    nsteps = len(input_mat) # nsteps for integrator
    timepoints = np.linspace(t0, tf, nsteps)
    params_obj_neural['interpolate_input'] = interpolate.interp1d(timepoints, input_mat, axis=0, kind = interp_kind_input, fill_value = "extrapolate")
    progress_milestones = np.linspace(0, nsteps, 10).astype('int')

    """ define the connectivity """

    modify_Connectome(ablation_mask, ablation_type)

    """ Define initial Condition """

    if type(custom_initcond) == bool:

        initcond = compute_initcond_neural()

    else:

        initcond = custom_initcond
        print("using the custom initial condition")

    """ Configuring the ODE Solver """
    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

        r = integrate.ode(membrane_voltageRHS_dyninput, compute_jacobian_dyninput).set_integrator('vode', atol = 1e-3, method = 'bdf')

    else:

        r = integrate.ode(membrane_voltageRHS_dyninput).set_integrator('vode', atol = 1e-3, method = 'bdf', with_jacobian = True)

    r.set_initial_value(initcond, t0)

    """ Additional Python step to store the trajectories """
    t = np.zeros(nsteps)
    traj = np.zeros((nsteps, params_obj_neural['N']))
    s_traj = np.zeros((nsteps, params_obj_neural['N']))
    vthmat = np.zeros((nsteps, params_obj_neural['N']))

    t[0] = t0
    traj[0, :] = initcond[:params_obj_neural['N']]
    s_traj[0, :] = initcond[params_obj_neural['N']:]
    vthmat[0, :] = EffVth_rhs(input_mat[0, :])

    print("Network integration prep completed...")

    """ Integrate the ODE(s) across each delta_t timestep """
    print("Computing network dynamics...")

    k = 1

    while r.successful() and k < nsteps:

        r.integrate(r.t + dt)

        t[k] = r.t
        traj[k, :] = r.y[:params_obj_neural['N']]
        s_traj[k, :] = r.y[params_obj_neural['N']:]
        vthmat[k, :] = EffVth_rhs(input_mat[k, :])

        k += 1

        if verbose == True:

            if k in progress_milestones:

                print(str(np.round((float(k) / nsteps) * 100, 1)) + '% ' + 'completed')

    result_dict_network = {
            "t": t,
            "dt": dt,
            "steps": nsteps,
            "raw_v_solution": traj,
            "s_solution": s_traj,
            "v_threshold": vthmat,
            "v_solution" : voltage_filter(np.subtract(traj, vthmat), 200, 1)
            }

    return result_dict_network

#########################################################################################################################################################################
### MASTER EXECUTION FUNCTIONS (ADVANCED) ###############################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

def run_network_externalV(input_mat, ext_voltage_mat, ablation_mask, \
    t_delta = 0.01, custom_initcond = False, ablation_type = "all", \
    interp_kind_input = 'linear', interp_kind_voltage = 'linear', verbose = True):
    
    assert 'params_obj_neural' in globals(), "Neural parameters and connectivity must be initialized before running the simulation"
    assert len(input_mat) == len(ext_voltage_mat), "Length of input_mat and ext_voltage_mat should be identical"
    # Also Check dimensions of input_mat and ext_voltage_mat

    t0 = 0
    tf = (len(input_mat) - 1) * t_delta
    dt = t_delta

    nsteps = len(input_mat)

    timepoints = np.linspace(t0, tf, nsteps)

    params_obj_neural['interpolate_voltage'] = interpolate.interp1d(timepoints, ext_voltage_mat, axis=0, kind = interp_kind_voltage, fill_value = "extrapolate")
    params_obj_neural['interpolate_input'] = interpolate.interp1d(timepoints, input_mat, axis=0, kind = interp_kind_input, fill_value = "extrapolate")
    progress_milestones = np.linspace(0, nsteps, 10).astype('int')

    """ define the connectivity """

    modify_Connectome(ablation_mask, ablation_type)

    """ Define initial condition """

    if type(custom_initcond) == bool:

        initcond = compute_initcond_neural()

    else:

        initcond = custom_initcond
        print("using the custom initial condition")

    """ Configuring the ODE Solver """
    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

        r = integrate.ode(membrane_voltageRHS_vext, compute_jacobian_vext).set_integrator('vode', atol = 1e-3, method = 'bdf')

    else:

        r = integrate.ode(membrane_voltageRHS_vext).set_integrator('vode', atol = 1e-3, method = 'bdf', with_jacobian = True)

    r.set_initial_value(initcond, t0)

    """ Additional Python step to store the trajectories """
    t = np.zeros(nsteps)
    traj = np.zeros((nsteps, params_obj_neural['N']))
    s_traj = np.zeros((nsteps, params_obj_neural['N']))
    vthmat = np.zeros((nsteps, params_obj_neural['N']))

    t[0] = t0
    traj[0, :] = initcond[:params_obj_neural['N']]
    s_traj[0, :] = initcond[params_obj_neural['N']:]
    vthmat[0, :] = EffVth_rhs(input_mat[0, :])

    print("Network integration prep completed...")

    """ Integrate the ODE(s) across each delta_t timestep """
    print("Computing network dynamics...")
    k = 1

    while r.successful() and k < nsteps:

        r.integrate(r.t + dt)

        t[k] = r.t
        traj[k, :] = r.y[:params_obj_neural['N']]
        s_traj[k, :] = r.y[params_obj_neural['N']:]
        vthmat[k, :] = EffVth_rhs(input_mat[k, :])

        k += 1

        if verbose == True:

            if k in progress_milestones:

                print(str(np.round((float(k) / nsteps) * 100, 1)) + '% ' + 'completed')

    result_dict_network = {
            "t": t,
            "dt": dt,
            "steps": nsteps,
            "raw_v_solution": traj,
            "s_solution": s_traj,
            "v_threshold": vthmat,
            "v_solution" : voltage_filter(np.subtract(traj, vthmat), 200, 1)
            }

    return result_dict_network

def run_network_fdb(input_mat, ablation_mask, \
    t_delta = 0.01, custom_initcond = False, ablation_type = "all", \
    interp_kind_input = 'linear', \
    custom_muscle_map = False, fdb_init = 1.38, t_delay = 0.54, reaction_scaling = 1, verbose = True):

    assert 'params_obj_neural' in globals(), "Neural parameters and connectivity must be initialized before running the simulation"

    t0 = 0
    tf = (len(input_mat) - 1) * t_delta
    dt = t_delta

    nsteps = len(input_mat)
    timepoints = np.linspace(t0, tf, nsteps)

    params_obj_neural['interpolate_input'] = interpolate.interp1d(timepoints, input_mat, axis=0, kind = interp_kind_input, fill_value = "extrapolate")
    params_obj_neural['fdb_init'] = fdb_init
    params_obj_neural['t_delay'] = t_delay
    params_obj_neural['reaction_scaling'] = reaction_scaling
    progress_milestones = np.linspace(0, nsteps, 10).astype('int')

    """ define the connectivity """

    modify_Connectome(ablation_mask, ablation_type)

    """ Calculate initial condition """

    if type(custom_initcond) == bool:

        initcond = compute_initcond_neural()

    else:

        initcond = custom_initcond
        print("using the custom initial condition")

    """ Configure muscle map for feedback """
    if custom_muscle_map == False:

        params_obj_neural['muscle_map'] = b_params.muscle_map_f
        params_obj_neural['muscle_map_pseudoinv'] = b_params.muscle_map_pseudoinv_f

    else:

        params_obj_neural['muscle_map'] = custom_muscle_map
        params_obj_neural['muscle_map_pseudoinv'] = np.linalg.pinv(custom_muscle_map)
        print("using the custom muscle map")

    """ Configuring the ODE Solver """
    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

        r = integrate.ode(membrane_voltageRHS_fdb, compute_jacobian_fdb).set_integrator('vode', atol = 1e-3, method = 'bdf')

    else:

        r = integrate.ode(membrane_voltageRHS_fdb).set_integrator('vode', atol = 1e-3, method = 'bdf', with_jacobian = True)

    r.set_initial_value(initcond, t0)

    """ Additional Python step to store the trajectories """
    t = np.zeros(nsteps)
    traj = np.zeros((nsteps, params_obj_neural['N']))
    vthmat = np.zeros((nsteps, params_obj_neural['N']))

    params_obj_neural['interpolate_traj'] = interpolate.interp1d(timepoints, traj, axis=0, fill_value = "extrapolate")
    params_obj_neural['interpolate_vthmat'] = interpolate.interp1d(timepoints, vthmat, axis=0, fill_value = "extrapolate")

    t[0] = t0
    traj[0, :] = initcond[:params_obj_neural['N']]
    vthmat[0, :] = EffVth_rhs(input_mat[0, :])

    print("Network integration prep completed...")

    """ Integrate the ODE(s) across each delta_t timestep """
    print("Computing network dynamics...")
    k = 1

    while r.successful() and k < nsteps:

        r.integrate(r.t + dt)

        t[k] = r.t
        traj[k, :] = r.y[:params_obj_neural['N']]
        vthmat[k, :] = EffVth_rhs(input_mat[k, :])

        params_obj_neural['interpolate_traj'] = interpolate.interp1d(timepoints, traj, axis=0, fill_value = "extrapolate")
        params_obj_neural['interpolate_vthmat'] = interpolate.interp1d(timepoints, vthmat, axis=0, fill_value = "extrapolate")

        k += 1

        if verbose == True:

            if k in progress_milestones:

                print(str(np.round((float(k) / nsteps) * 100, 1)) + '% ' + 'completed')

    result_dict_network = {
            "t": t,
            "dt": dt,
            "steps": nsteps,
            "raw_v_solution": traj,
            "v_threshold": vthmat,
            "v_solution" : voltage_filter(np.subtract(traj, vthmat), 200, 1)
            }

    return result_dict_network

def run_network_fdb_externalV(input_mat, ext_voltage_mat, ablation_mask, \
    t_delta = 0.01, custom_initcond = False, ablation_type = "all", \
    interp_kind_input = 'linear', interp_kind_voltage = 'linear', \
    custom_muscle_map = False, fdb_init = 1.38, t_delay = 0.54, reaction_scaling = 1, verbose = True):

    assert 'params_obj_neural' in globals(), "Neural parameters and connectivity must be initialized before running the simulation"
    assert len(input_mat) == len(ext_voltage_mat), "Length of input_mat and ext_voltage_mat should be identical"

    t0 = 0
    tf = (len(input_mat) - 1) * t_delta
    dt = t_delta

    inmask = np.zeros(params_obj_neural['N'])
    nsteps = len(input_mat)
    timepoints = np.linspace(t0, tf, nsteps)

    params_obj_neural['interpolate_voltage'] = interpolate.interp1d(timepoints, ext_voltage_mat, axis=0, kind = interp_kind_voltage, fill_value = "extrapolate")
    params_obj_neural['interpolate_input'] = interpolate.interp1d(timepoints, input_mat, axis=0, kind = interp_kind_input, fill_value = "extrapolate")

    params_obj_neural['fdb_init'] = fdb_init
    params_obj_neural['t_delay'] = t_delay
    params_obj_neural['reaction_scaling'] = reaction_scaling
    progress_milestones = np.linspace(0, nsteps, 10).astype('int')

    """ define the connectivity """

    modify_Connectome(ablation_mask, ablation_type)

    """ Define initial condition """

    if type(custom_initcond) == bool:

        initcond = compute_initcond_neural()

    else:

        initcond = custom_initcond
        print("using the custom initial condition")

    """ Configure muscle map for feedback """
    if custom_muscle_map == False:

        params_obj_neural['muscle_map'] = b_params.muscle_map_f
        params_obj_neural['muscle_map_pseudoinv'] = b_params.muscle_map_pseudoinv_f

    else:

        params_obj_neural['muscle_map'] = custom_muscle_map
        params_obj_neural['muscle_map_pseudoinv'] = np.linalg.pinv(custom_muscle_map)
        print("using the custom muscle map")

    """ Configuring the ODE Solver """
    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

        r = integrate.ode(membrane_voltageRHS_fdb_vext, compute_jacobian_fdb_vext).set_integrator('vode', atol = 1e-3, method = 'bdf')

    else:

        r = integrate.ode(membrane_voltageRHS_fdb_vext).set_integrator('vode', atol = 1e-3, method = 'bdf', with_jacobian = True)

    r.set_initial_value(initcond, t0)

    """ Additional Python step to store the trajectories """
    t = np.zeros(nsteps)
    traj = np.zeros((nsteps, params_obj_neural['N']))
    s_traj = np.zeros((nsteps, params_obj_neural['N']))
    vthmat = np.zeros((nsteps, params_obj_neural['N']))

    params_obj_neural['interpolate_traj'] = interpolate.interp1d(timepoints, traj, axis=0, fill_value = "extrapolate")
    params_obj_neural['interpolate_vthmat'] = interpolate.interp1d(timepoints, vthmat, axis=0, fill_value = "extrapolate")

    t[0] = t0
    traj[0, :] = initcond[:params_obj_neural['N']]
    s_traj[0, :] = initcond[params_obj_neural['N']:]
    vthmat[0, :] = EffVth_rhs(input_mat[0, :])

    print("Network integration prep completed...")

    """ Integrate the ODE(s) across each delta_t timestep """
    print("Computing network dynamics...")
    k = 1

    while r.successful() and k < nsteps:

        r.integrate(r.t + dt)

        t[k] = r.t
        traj[k, :] = r.y[:params_obj_neural['N']]
        s_traj[k, :] = r.y[params_obj_neural['N']:]
        vthmat[k, :] = EffVth_rhs(input_mat[k, :])

        params_obj_neural['interpolate_traj'] = interpolate.interp1d(timepoints, traj, axis=0, fill_value = "extrapolate")
        params_obj_neural['interpolate_vthmat'] = interpolate.interp1d(timepoints, vthmat, axis=0, fill_value = "extrapolate")

        k += 1

        if verbose == True:

            if k in progress_milestones:

                print(str(np.round((float(k) / nsteps) * 100, 1)) + '% ' + 'completed')

    result_dict_network = {
            "t": t,
            "dt": dt,
            "steps": nsteps,
            "raw_v_solution": traj,
            "s_solution": s_traj,
            "v_threshold": vthmat,
            "v_solution" : voltage_filter(np.subtract(traj, vthmat), 200, 1)
            }

    return result_dict_network

########################################################################################################################################################################
### SIMULATION SPECIFIC ENVIRONMENT FUNCTIONS ##########################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

def compute_initcond_neural():

    AWA_nonlinear_bool = params_obj_neural['nonlinear_AWA']
    AVL_nonlinear_bool = params_obj_neural['nonlinear_AVL']

    # Baseline

    if AWA_nonlinear_bool == False and AVL_nonlinear_bool == False:

        voltage_initcond = 10**(-4)*np.random.normal(0, 0.94, params_obj_neural['N'])
        synaptic_initcond = 10**(-4)*np.random.normal(0, 0.94, params_obj_neural['N'])
        full_initcond = np.concatenate([voltage_initcond, synaptic_initcond])

    # AWA nonlinear only

    elif AWA_nonlinear_bool == True and AVL_nonlinear_bool == False:

        params_obj_neural['AWA_nonlinear_params'] = n_params.AWA_nonlinear_params.copy()

        voltage_initcond = 10**(-4)*np.random.normal(0, 0.94, params_obj_neural['N'])
        np.put(voltage_initcond, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], -74.99)
        synaptic_initcond = 10**(-4)*np.random.normal(0, 0.94, params_obj_neural['N'])

        awa_initcond1 = np.zeros(params_obj_neural['N'],) 
        np.put(awa_initcond1, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 2.05e-7)
        awa_initcond2 = np.zeros(params_obj_neural['N'],)
        np.put(awa_initcond2, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 9.71e-6)
        awa_initcond3 = np.zeros(params_obj_neural['N'],)
        np.put(awa_initcond3, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 9.71e-6)
        awa_initcond4 = np.zeros(params_obj_neural['N'],)
        np.put(awa_initcond4, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 8.35e-7)
        awa_initcond5 = np.zeros(params_obj_neural['N'],)
        np.put(awa_initcond5, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 0.000151)
        awa_initcond6 = np.zeros(params_obj_neural['N'],)
        np.put(awa_initcond6, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 2.069e-9)
        awa_initcond7 = np.zeros(params_obj_neural['N'],)
        np.put(awa_initcond7, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 1.7e-10)

        full_initcond = np.concatenate([voltage_initcond, synaptic_initcond, awa_initcond1, awa_initcond2, awa_initcond3, awa_initcond4,
         awa_initcond5, awa_initcond6, awa_initcond7])

    # AVL nonlinear only

    elif AWA_nonlinear_bool == False and AVL_nonlinear_bool == True:

        params_obj_neural['AVL_nonlinear_params'] = n_params.AVL_nonlinear_params.copy()

        voltage_initcond = 10**(-4)*np.random.normal(0, 0.94, params_obj_neural['N'])
        np.put(voltage_initcond, params_obj_neural['AVL_nonlinear_params']['AVL_ind'], -45.7)
        synaptic_initcond = 10**(-4)*np.random.normal(0, 0.94, params_obj_neural['N'])
        avl_initcond = np.zeros(params_obj_neural['N'] * 16,)

        full_initcond = np.concatenate([voltage_initcond, synaptic_initcond, avl_initcond])

    # AWA + AVL nonlinear

    else:

        params_obj_neural['AWA_nonlinear_params'] = n_params.AWA_nonlinear_params.copy()
        params_obj_neural['AVL_nonlinear_params'] = n_params.AVL_nonlinear_params.copy()

        voltage_initcond = 10**(-4)*np.random.normal(0, 0.94, params_obj_neural['N'])
        np.put(voltage_initcond, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], -74.99)
        np.put(voltage_initcond, params_obj_neural['AVL_nonlinear_params']['AVL_ind'], -45.7)
        synaptic_initcond = 10**(-4)*np.random.normal(0, 0.94, params_obj_neural['N'])

        awa_initcond1 = np.zeros(params_obj_neural['N'],) 
        np.put(awa_initcond1, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 2.05e-7)
        awa_initcond2 = np.zeros(params_obj_neural['N'],)
        np.put(awa_initcond2, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 9.71e-6)
        awa_initcond3 = np.zeros(params_obj_neural['N'],)
        np.put(awa_initcond3, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 9.71e-6)
        awa_initcond4 = np.zeros(params_obj_neural['N'],)
        np.put(awa_initcond4, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 8.35e-7)
        awa_initcond5 = np.zeros(params_obj_neural['N'],)
        np.put(awa_initcond5, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 0.000151)
        awa_initcond6 = np.zeros(params_obj_neural['N'],)
        np.put(awa_initcond6, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 2.069e-9)
        awa_initcond7 = np.zeros(params_obj_neural['N'],)
        np.put(awa_initcond7, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], 1.7e-10)

        avl_initcond = np.zeros(params_obj_neural['N'] * 16,)

        full_initcond = np.concatenate([voltage_initcond, synaptic_initcond, awa_initcond1, awa_initcond2, awa_initcond3, awa_initcond4,
         awa_initcond5, awa_initcond6, awa_initcond7, avl_initcond])

    return full_initcond

def EffVth(Gg, Gs):

    Gcmat = np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    EcVec = np.multiply(params_obj_neural['Ec'], np.ones((params_obj_neural['N'], 1)))

    M1 = -Gcmat
    b1 = np.multiply(params_obj_neural['Gc'], EcVec)

    Ggap = np.multiply(params_obj_neural['ggap'], Gg)
    Ggapdiag = np.subtract(Ggap, np.diag(np.diag(Ggap)))
    Ggapsum = Ggapdiag.sum(axis = 1)
    Ggapsummat = sparse.spdiags(Ggapsum, 0, params_obj_neural['N'], params_obj_neural['N']).toarray()
    M2 = -np.subtract(Ggapsummat, Ggapdiag)

    Gs_ij = np.multiply(params_obj_neural['gsyn'], Gs)
    s_eq = round((params_obj_neural['ar']/(params_obj_neural['ar'] + 2 * params_obj_neural['ad'])), 4)
    sjmat = np.multiply(s_eq, np.ones((params_obj_neural['N'], params_obj_neural['N'])))
    S_eq = np.multiply(s_eq, np.ones((params_obj_neural['N'], 1)))
    Gsyn = np.multiply(sjmat, Gs_ij)
    Gsyndiag = np.subtract(Gsyn, np.diag(np.diag(Gsyn)))
    Gsynsum = Gsyndiag.sum(axis = 1)
    M3 = -sparse.spdiags(Gsynsum, 0, params_obj_neural['N'], params_obj_neural['N']).toarray()

    #b3 = np.dot(Gs_ij, np.multiply(s_eq, params_obj_neural['E']))
    b3 = np.dot(np.multiply(Gs_ij, params_obj_neural['EMat']), s_eq * np.ones((params_obj_neural['N'], 1)))

    M = M1 + M2 + M3

    (P, LL, UU) = linalg.lu(M)
    bbb = -b1 - b3
    bb = np.reshape(bbb, params_obj_neural['N'])

    params_obj_neural['LL'] = LL
    params_obj_neural['UU'] = UU
    params_obj_neural['bb'] = bb

def EffVth_rhs(inmask):

    InputMask = np.multiply(params_obj_neural['iext'], inmask)
    b = np.subtract(params_obj_neural['bb'], InputMask)

    vth = linalg.solve_triangular(params_obj_neural['UU'], linalg.solve_triangular(params_obj_neural['LL'], b, lower = True, check_finite=False), check_finite=False)

    return vth

def modify_Connectome(ablation_mask, ablation_type):

    # ablation_type can be 'all': ablate both synaptic and gap junctions, 'syn': Synaptic only and 'gap': Gap junctions only

    if np.sum(ablation_mask) == params_obj_neural['N']:

        apply_Mat = np.ones((params_obj_neural['N'], params_obj_neural['N']))

        params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat)
        params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat)

        print("All neurons are healthy")

        EffVth(params_obj_neural['Gg_Dynamic'], params_obj_neural['Gs_Dynamic'])

    else:

        apply_Col = np.tile(ablation_mask, (params_obj_neural['N'], 1))
        apply_Row = np.transpose(apply_Col)

        apply_Mat = np.multiply(apply_Col, apply_Row)

        if ablation_type == "all":

            params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat)
            params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat)

            #print("Ablating both Gap and Syn")

        elif ablation_type == "syn":

            params_obj_neural['Gg_Dynamic'] = params_obj_neural['Gg_Static'].copy()
            params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat)

            print("Ablating only Syn")

        elif ablation_type == "gap":

            params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat)
            params_obj_neural['Gs_Dynamic'] = params_obj_neural['Gs_Static'].copy()

            print("Ablating only Gap")

        EffVth(params_obj_neural['Gg_Dynamic'], params_obj_neural['Gs_Dynamic'])


def ablate_edges(neurons_from, neurons_to, conn_type):

    apply_Mat = np.ones((params_obj_neural['N'],params_obj_neural['N']), dtype = 'bool')
    apply_Mat_Identity = np.ones((params_obj_neural['N'],params_obj_neural['N']), dtype = 'bool')

    for k in range(len(neurons_from)):

        neuron_from_ind = []
        neurons_target_inds = []

        neuron_from = neurons_from[k]
        neurons_target = neurons_to[k]

        neuron_from_ind.append(neuron_names.index(neuron_from))

        for neuron_target in neurons_target:

            neurons_target_inds.append(neuron_names.index(neuron_target))

        if conn_type == 'syn':

            apply_Mat[neurons_target_inds, neuron_from_ind] = 0

        elif conn_type == 'gap':

            apply_Mat[neurons_target_inds, neuron_from_ind] = 0
            apply_Mat[neuron_from_ind, neurons_target_inds] = 0

    if conn_type == 'syn':

        params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat_Identity)
        params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat)

    elif conn_type == 'gap':

        params_obj_neural['Gg_Dynamic'] = np.multiply(params_obj_neural['Gg_Static'], apply_Mat)
        params_obj_neural['Gs_Dynamic'] = np.multiply(params_obj_neural['Gs_Static'], apply_Mat_Identity)

    EffVth(params_obj_neural['Gg_Dynamic'], params_obj_neural['Gs_Dynamic'])

def add_gap_junctions(neuron_pairs_mat, gap_weights_vec):

    """ neuron_pairs_mat is (N, 2) numpy.array form where N is the total number of pairs. 
        Each element should be of type float denoting the index number of neuron"""

    """ gap_weights_vec is (N,) numpy.array form where N is the total number of pair.
        Each element should be of type float denoting the gap weights to be added for each pair"""

    """ This function should be executed after modify_Connectome """

    num_pairs = len(neuron_pairs_mat)

    for k in range(num_pairs):

        neuron_ind_1 = neuron_pairs_mat[k, 0]
        neuron_ind_2 - neuron_pairs_mat[k, 1]

        params_obj_neural['Gg_Dynamic'][neuron_ind_1, neuron_ind_2] = params_obj_neural['Gg_Dynamic'][neuron_ind_1, neuron_ind_2] + gap_weights_vec[k]
        params_obj_neural['Gg_Dynamic'][neuron_ind_2, neuron_ind_1] = params_obj_neural['Gg_Dynamic'][neuron_ind_2, neuron_ind_1] + gap_weights_vec[k]

    EffVth(params_obj_neural['Gg_Dynamic'], params_obj_neural['Gs_Dynamic'])

def voltage_filter(v_vec, vmax, scaler):
    
    filtered = vmax * np.tanh(scaler * np.divide(v_vec, vmax))
    
    return filtered

def infer_force_voltage(v_vec, muscle_map, muscle_map_pseudoinv):

    muscle_input = np.dot(muscle_map, v_vec)

    reaction_scaling_mask = np.ones(muscle_input.shape)
    reaction_scaling_mask[muscle_input > 0] = params_obj_neural['reaction_scaling']
    muscle_input = np.multiply(muscle_input, reaction_scaling_mask)

    inferred_v = np.dot(muscle_map_pseudoinv, muscle_input)

    return inferred_v

#########################################################################################################################################################################
### RIGHT HAND SIDE FUNCTIONS + JACOBIAN ################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

def membrane_voltageRHS_constinput(t, y):

    Vvec, SVec = np.split(y[:params_obj_neural['N']*2], 2)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(params_obj_neural['Gg_Dynamic'], np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, params_obj_neural['vth'])))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], params_obj_neural['inmask'])

    """ dV and dS and merge them back to dydt """
    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

        dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
        dS = np.subtract(SynRise, SynDrop)

        return np.concatenate((dV, dS))

    else:

        baseline_dict = {

        "Vvec": Vvec.copy(),
        "y": y.copy(),
        "VsubEc": VsubEc.copy(),
        "GapCon": GapCon.copy(),
        "SynapCon": SynapCon.copy(),
        "Input": Input.copy(),
        "SynRise": SynRise.copy(),
        "SynDrop": SynDrop.copy()

        }

        d_network = combine_baseline_nonlinear_currents(baseline_dict)

        return d_network

def compute_jacobian_constinput(t, y):

    Vvec, SVec = np.split(y, 2)
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn'], params_obj_neural['Gs_Dynamic'])
    J1_M3 = np.diag(np.dot(-Gsyn, SVec))

    J1 = (J1_M1 + J1_M2 + J1_M3) / params_obj_neural['C']

    J2_M4_2 = np.subtract(params_obj_neural['EMat'], np.transpose(Vrep))
    J2 = np.multiply(Gsyn, J2_M4_2) / params_obj_neural['C']

    sigmoid_V = np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, params_obj_neural['vth']))))
    J3_1 = np.multiply(params_obj_neural['ar'], 1 - SVec)
    J3_2 = np.multiply(params_obj_neural['B'], sigmoid_V)
    J3_3 = 1 - sigmoid_V
    J3 = np.diag(np.multiply(np.multiply(J3_1, J3_2), J3_3))

    J4 = np.diag(np.subtract(np.multiply(-params_obj_neural['ar'], sigmoid_V), params_obj_neural['ad']))

    J_row1 = np.hstack((J1, J2))
    J_row2 = np.hstack((J3, J4))
    J = np.vstack((J_row1, J_row2))

    return J

def membrane_voltageRHS_dyninput(t, y):

    Vvec, SVec = np.split(y[:params_obj_neural['N']*2], 2)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(params_obj_neural['Gg_Dynamic'], np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ interpolate input """
    inmask = params_obj_neural['interpolate_input'](t)
    vth = EffVth_rhs(inmask)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth)))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], inmask)

    """ dV and dS and merge them back to dydt """
    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

        dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
        dS = np.subtract(SynRise, SynDrop)

        return np.concatenate((dV, dS))

    else:

        baseline_dict = {

        "Vvec": Vvec.copy(),
        "y": y.copy(),
        "VsubEc": VsubEc.copy(),
        "GapCon": GapCon.copy(),
        "SynapCon": SynapCon.copy(),
        "Input": Input.copy(),
        "SynRise": SynRise.copy(),
        "SynDrop": SynDrop.copy()

        }

        d_network = combine_baseline_nonlinear_currents(baseline_dict)

        return d_network

def compute_jacobian_dyninput(t, y):

    Vvec, SVec = np.split(y, 2)
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn'], params_obj_neural['Gs_Dynamic'])
    J1_M3 = np.diag(np.dot(-Gsyn, SVec))

    J1 = (J1_M1 + J1_M2 + J1_M3) / params_obj_neural['C']

    J2_M4_2 = np.subtract(params_obj_neural['EMat'], np.transpose(Vrep))
    J2 = np.multiply(Gsyn, J2_M4_2) / params_obj_neural['C']

    inmask = params_obj_neural['interpolate_input'](t)
    vth = EffVth_rhs(inmask)

    sigmoid_V = np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth))))
    J3_1 = np.multiply(params_obj_neural['ar'], 1 - SVec)
    J3_2 = np.multiply(params_obj_neural['B'], sigmoid_V)
    J3_3 = 1 - sigmoid_V
    J3 = np.diag(np.multiply(np.multiply(J3_1, J3_2), J3_3))

    J4 = np.diag(np.subtract(np.multiply(-params_obj_neural['ar'], sigmoid_V), params_obj_neural['ad']))

    J_row1 = np.hstack((J1, J2))
    J_row2 = np.hstack((J3, J4))
    J = np.vstack((J_row1, J_row2))

    return J

def membrane_voltageRHS_vext(t, y):

    Vvec, SVec = np.split(y[:params_obj_neural['N']*2], 2)

    v_ext = params_obj_neural['interpolate_voltage'](t)
    Vvec = np.add(Vvec, v_ext)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(params_obj_neural['Gg_Dynamic'], np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ interpolate input """
    inmask = params_obj_neural['interpolate_input'](t)
    vth = EffVth_rhs(inmask)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth)))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], inmask)

    """ dV and dS and merge them back to dydt """
    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

        dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
        dS = np.subtract(SynRise, SynDrop)

        return np.concatenate((dV, dS))

    else:

        baseline_dict = {

        "Vvec": Vvec.copy(),
        "y": y.copy(),
        "VsubEc": VsubEc.copy(),
        "GapCon": GapCon.copy(),
        "SynapCon": SynapCon.copy(),
        "Input": Input.copy(),
        "SynRise": SynRise.copy(),
        "SynDrop": SynDrop.copy()

        }

        d_network = combine_baseline_nonlinear_currents(baseline_dict)

        return d_network

def compute_jacobian_vext(t, y):

    Vvec, SVec = np.split(y, 2)
    v_ext = params_obj_neural['interpolate_voltage'](t)
    Vvec = np.add(Vvec, v_ext)
    
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn'], params_obj_neural['Gs_Dynamic'])
    J1_M3 = np.diag(np.dot(-Gsyn, SVec))

    J1 = (J1_M1 + J1_M2 + J1_M3) / params_obj_neural['C']

    J2_M4_2 = np.subtract(params_obj_neural['EMat'], np.transpose(Vrep))
    J2 = np.multiply(Gsyn, J2_M4_2) / params_obj_neural['C']

    inmask = params_obj_neural['interpolate_input'](t)
    vth = EffVth_rhs(inmask)

    sigmoid_V = np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth))))
    J3_1 = np.multiply(params_obj_neural['ar'], 1 - SVec)
    J3_2 = np.multiply(params_obj_neural['B'], sigmoid_V)
    J3_3 = 1 - sigmoid_V
    J3 = np.diag(np.multiply(np.multiply(J3_1, J3_2), J3_3))

    J4 = np.diag(np.subtract(np.multiply(-params_obj_neural['ar'], sigmoid_V), params_obj_neural['ad']))

    J_row1 = np.hstack((J1, J2))
    J_row2 = np.hstack((J3, J4))
    J = np.vstack((J_row1, J_row2))

    return J

def membrane_voltageRHS_fdb(t, y):

    Vvec, SVec = np.split(y[:params_obj_neural['N']*2], 2)

    if t > params_obj_neural['fdb_init']:

        delayed_v = params_obj_neural['interpolate_traj'](t - params_obj_neural['t_delay'])
        delayed_vth = params_obj_neural['interpolate_vthmat'](t - params_obj_neural['t_delay'])
        delayed_vsubvth = np.subtract(delayed_v, delayed_vth)
        delayed_vsub = infer_force_voltage(delayed_vsubvth, params_obj_neural['muscle_map'], params_obj_neural['muscle_map_pseudoinv']) 

        Vvec = np.add(Vvec, delayed_vsub)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(params_obj_neural['Gg_Dynamic'], np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ interpolate input """
    inmask = params_obj_neural['interpolate_input'](t)
    vth = EffVth_rhs(inmask)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth)))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], inmask)

    """ dV and dS and merge them back to dydt """
    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

        dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
        dS = np.subtract(SynRise, SynDrop)

        return np.concatenate((dV, dS))

    else:

        baseline_dict = {

        "Vvec": Vvec.copy(),
        "y": y.copy(),
        "VsubEc": VsubEc.copy(),
        "GapCon": GapCon.copy(),
        "SynapCon": SynapCon.copy(),
        "Input": Input.copy(),
        "SynRise": SynRise.copy(),
        "SynDrop": SynDrop.copy()

        }

        d_network = combine_baseline_nonlinear_currents(baseline_dict)

        return d_network

def compute_jacobian_fdb(t, y):

    Vvec, SVec = np.split(y, 2)

    if t > params_obj_neural['fdb_init']:

        delayed_v = params_obj_neural['interpolate_traj'](t - params_obj_neural['t_delay'])
        delayed_vth = params_obj_neural['interpolate_vthmat'](t - params_obj_neural['t_delay'])
        delayed_vsubvth = np.subtract(delayed_v, delayed_vth)
        delayed_vsub = infer_force_voltage(delayed_vsubvth, params_obj_neural['muscle_map'], params_obj_neural['muscle_map_pseudoinv']) 

        Vvec = np.add(Vvec, delayed_vsub)
    
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn'], params_obj_neural['Gs_Dynamic'])
    J1_M3 = np.diag(np.dot(-Gsyn, SVec))

    J1 = (J1_M1 + J1_M2 + J1_M3) / params_obj_neural['C']

    J2_M4_2 = np.subtract(params_obj_neural['EMat'], np.transpose(Vrep))
    J2 = np.multiply(Gsyn, J2_M4_2) / params_obj_neural['C']

    inmask = params_obj_neural['interpolate_input'](t)
    vth = EffVth_rhs(inmask)

    sigmoid_V = np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth))))
    J3_1 = np.multiply(params_obj_neural['ar'], 1 - SVec)
    J3_2 = np.multiply(params_obj_neural['B'], sigmoid_V)
    J3_3 = 1 - sigmoid_V
    J3 = np.diag(np.multiply(np.multiply(J3_1, J3_2), J3_3))

    J4 = np.diag(np.subtract(np.multiply(-params_obj_neural['ar'], sigmoid_V), params_obj_neural['ad']))

    J_row1 = np.hstack((J1, J2))
    J_row2 = np.hstack((J3, J4))
    J = np.vstack((J_row1, J_row2))

    return J

def membrane_voltageRHS_fdb_vext(t, y):

    Vvec, SVec = np.split(y[:params_obj_neural['N']*2], 2)
    v_ext = params_obj_neural['interpolate_voltage'](t)
    Vvec = np.add(Vvec, v_ext)

    if t > params_obj_neural['fdb_init']:

        delayed_v = params_obj_neural['interpolate_traj'](t - params_obj_neural['t_delay'])
        delayed_vth = params_obj_neural['interpolate_vthmat'](t - params_obj_neural['t_delay'])
        delayed_vsubvth = np.subtract(delayed_v, delayed_vth)
        delayed_vsub = infer_force_voltage(delayed_vsubvth, params_obj_neural['muscle_map'], params_obj_neural['muscle_map_pseudoinv']) 

        Vvec = np.add(Vvec, delayed_vsub)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(params_obj_neural['Gg_Dynamic'], np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ interpolate input """
    inmask = params_obj_neural['interpolate_input'](t)
    vth = EffVth_rhs(inmask)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth)))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], inmask)

    """ dV and dS and merge them back to dydt """
    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

        dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
        dS = np.subtract(SynRise, SynDrop)

        return np.concatenate((dV, dS))

    else:

        baseline_dict = {

        "Vvec": Vvec.copy(),
        "y": y.copy(),
        "VsubEc": VsubEc.copy(),
        "GapCon": GapCon.copy(),
        "SynapCon": SynapCon.copy(),
        "Input": Input.copy(),
        "SynRise": SynRise.copy(),
        "SynDrop": SynDrop.copy()

        }

        d_network = combine_baseline_nonlinear_currents(baseline_dict)

        return d_network

def compute_jacobian_fdb_vext(t, y):

    Vvec, SVec = np.split(y, 2)
    v_ext = params_obj_neural['interpolate_voltage'](t)
    Vvec = np.add(Vvec, v_ext)

    if t > params_obj_neural['fdb_init']:

        delayed_v = params_obj_neural['interpolate_traj'](t - params_obj_neural['t_delay'])
        delayed_vth = params_obj_neural['interpolate_vthmat'](t - params_obj_neural['t_delay'])
        delayed_vsubvth = np.subtract(delayed_v, delayed_vth)
        delayed_vsub = infer_force_voltage(delayed_vsubvth, params_obj_neural['muscle_map'], params_obj_neural['muscle_map_pseudoinv']) 

        Vvec = np.add(Vvec, delayed_vsub)
    
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn'], params_obj_neural['Gs_Dynamic'])
    J1_M3 = np.diag(np.dot(-Gsyn, SVec))

    J1 = (J1_M1 + J1_M2 + J1_M3) / params_obj_neural['C']

    J2_M4_2 = np.subtract(params_obj_neural['EMat'], np.transpose(Vrep))
    J2 = np.multiply(Gsyn, J2_M4_2) / params_obj_neural['C']

    inmask = params_obj_neural['interpolate_input'](t)
    vth = EffVth_rhs(inmask)

    sigmoid_V = np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth))))
    J3_1 = np.multiply(params_obj_neural['ar'], 1 - SVec)
    J3_2 = np.multiply(params_obj_neural['B'], sigmoid_V)
    J3_3 = 1 - sigmoid_V
    J3 = np.diag(np.multiply(np.multiply(J3_1, J3_2), J3_3))

    J4 = np.diag(np.subtract(np.multiply(-params_obj_neural['ar'], sigmoid_V), params_obj_neural['ad']))

    J_row1 = np.hstack((J1, J2))
    J_row2 = np.hstack((J3, J4))
    J = np.vstack((J_row1, J_row2))

    return J

def membrane_voltageRHS_fdb_vext_spatialgrad(t, y):

    Vvec, SVec = np.split(y[:params_obj_neural['N']*2], 2)
    v_ext = params_obj_neural['interpolate_voltage'](t)
    Vvec = np.add(Vvec, v_ext)

    if t > params_obj_neural['fdb_init']:

        delayed_v = params_obj_neural['interpolate_traj'](t - params_obj_neural['t_delay'])
        delayed_vth = params_obj_neural['interpolate_vthmat'](t - params_obj_neural['t_delay'])
        delayed_vsubvth = np.subtract(delayed_v, delayed_vth)
        delayed_vsub = infer_force_voltage(delayed_vsubvth, params_obj_neural['muscle_map'], params_obj_neural['muscle_map_pseudoinv']) 

        Vvec = np.add(Vvec, delayed_vsub)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(params_obj_neural['Gg_Dynamic'], np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ interpolate vth """
    vth = EffVth_rhs(params_obj_neural['inmask'])

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth)))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], params_obj_neural['inmask'])

    """ dV and dS and merge them back to dydt """
    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

        dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
        dS = np.subtract(SynRise, SynDrop)

        return np.concatenate((dV, dS))

    else:

        baseline_dict = {

        "Vvec": Vvec.copy(),
        "y": y.copy(),
        "VsubEc": VsubEc.copy(),
        "GapCon": GapCon.copy(),
        "SynapCon": SynapCon.copy(),
        "Input": Input.copy(),
        "SynRise": SynRise.copy(),
        "SynDrop": SynDrop.copy()

        }

        d_network = combine_baseline_nonlinear_currents(baseline_dict)

        return d_network

def compute_jacobian_fdb_vext_spatialgrad(t, y):

    Vvec, SVec = np.split(y, 2)
    v_ext = params_obj_neural['interpolate_voltage'](t)
    Vvec = np.add(Vvec, v_ext)

    if t > params_obj_neural['fdb_init']:

        delayed_v = params_obj_neural['interpolate_traj'](t - params_obj_neural['t_delay'])
        delayed_vth = params_obj_neural['interpolate_vthmat'](t - params_obj_neural['t_delay'])
        delayed_vsubvth = np.subtract(delayed_v, delayed_vth)
        delayed_vsub = infer_force_voltage(delayed_vsubvth, params_obj_neural['muscle_map'], params_obj_neural['muscle_map_pseudoinv']) 

        Vvec = np.add(Vvec, delayed_vsub)
    
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn'], params_obj_neural['Gs_Dynamic'])
    J1_M3 = np.diag(np.dot(-Gsyn, SVec))

    J1 = (J1_M1 + J1_M2 + J1_M3) / params_obj_neural['C']

    J2_M4_2 = np.subtract(params_obj_neural['EMat'], np.transpose(Vrep))
    J2 = np.multiply(Gsyn, J2_M4_2) / params_obj_neural['C']

    vth = EffVth_rhs(params_obj_neural['inmask'])

    sigmoid_V = np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth))))
    J3_1 = np.multiply(params_obj_neural['ar'], 1 - SVec)
    J3_2 = np.multiply(params_obj_neural['B'], sigmoid_V)
    J3_3 = 1 - sigmoid_V
    J3 = np.diag(np.multiply(np.multiply(J3_1, J3_2), J3_3))

    J4 = np.diag(np.subtract(np.multiply(-params_obj_neural['ar'], sigmoid_V), params_obj_neural['ad']))

    J_row1 = np.hstack((J1, J2))
    J_row2 = np.hstack((J3, J4))
    J = np.vstack((J_row1, J_row2))

    return J

def membrane_voltageRHS_vext_spatialgrad(t, y):

    Vvec, SVec = np.split(y[:params_obj_neural['N']*2], 2)

    v_ext = params_obj_neural['interpolate_voltage'](t)
    Vvec = np.add(Vvec, v_ext)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(params_obj_neural['Gg_Dynamic'], np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ refer input """
    vth = EffVth_rhs(params_obj_neural['inmask'])

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth)))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], params_obj_neural['inmask'])

    """ dV and dS and merge them back to dydt """
    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

        dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
        dS = np.subtract(SynRise, SynDrop)

        return np.concatenate((dV, dS))

    else:

        baseline_dict = {

        "Vvec": Vvec.copy(),
        "y": y.copy(),
        "VsubEc": VsubEc.copy(),
        "GapCon": GapCon.copy(),
        "SynapCon": SynapCon.copy(),
        "Input": Input.copy(),
        "SynRise": SynRise.copy(),
        "SynDrop": SynDrop.copy()

        }

        d_network = combine_baseline_nonlinear_currents(baseline_dict)

        return d_network

def compute_jacobian_vext_spatialgrad(t, y):

    Vvec, SVec = np.split(y, 2)

    v_ext = params_obj_neural['interpolate_voltage'](t)
    Vvec = np.add(Vvec, v_ext)
    
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn'], params_obj_neural['Gs_Dynamic'])
    J1_M3 = np.diag(np.dot(-Gsyn, SVec))

    J1 = (J1_M1 + J1_M2 + J1_M3) / params_obj_neural['C']

    J2_M4_2 = np.subtract(params_obj_neural['EMat'], np.transpose(Vrep))
    J2 = np.multiply(Gsyn, J2_M4_2) / params_obj_neural['C']

    vth = EffVth_rhs(params_obj_neural['inmask'])

    sigmoid_V = np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth))))
    J3_1 = np.multiply(params_obj_neural['ar'], 1 - SVec)
    J3_2 = np.multiply(params_obj_neural['B'], sigmoid_V)
    J3_3 = 1 - sigmoid_V
    J3 = np.diag(np.multiply(np.multiply(J3_1, J3_2), J3_3))

    J4 = np.diag(np.subtract(np.multiply(-params_obj_neural['ar'], sigmoid_V), params_obj_neural['ad']))

    J_row1 = np.hstack((J1, J2))
    J_row2 = np.hstack((J3, J4))
    J = np.vstack((J_row1, J_row2))

    return J

########################## NON-LINEAR CURRENT #####################################################################################################################################################

def AWA_nonlinear_current(AWA_dyn_vec): # Incorporate leaky current

    AWA_dict = params_obj_neural['AWA_nonlinear_params']

    Vvec, wVec, c1Vec, c2Vec, bkVec, sloVec, slo2Vec, kbVec = np.split(AWA_dyn_vec, 8)

    """ channel variables """
    v = Vvec[params_obj_neural['AWA_nonlinear_params']['AWA_inds']]
    w = wVec[params_obj_neural['AWA_nonlinear_params']['AWA_inds']]
    c1 = c1Vec[params_obj_neural['AWA_nonlinear_params']['AWA_inds']]
    c2 = c2Vec[params_obj_neural['AWA_nonlinear_params']['AWA_inds']]
    bk = bkVec[params_obj_neural['AWA_nonlinear_params']['AWA_inds']]
    slo = sloVec[params_obj_neural['AWA_nonlinear_params']['AWA_inds']]
    slo2 = slo2Vec[params_obj_neural['AWA_nonlinear_params']['AWA_inds']]
    kb = kbVec[params_obj_neural['AWA_nonlinear_params']['AWA_inds']]

    """ potassium current """
    xinf = 0.5*(1 + np.tanh((v-params_obj_neural['AWA_nonlinear_params']['vk1'])/params_obj_neural['AWA_nonlinear_params']['vk2']))
    minf = 0.5*(1 + np.tanh((v-params_obj_neural['AWA_nonlinear_params']['vm1'])/params_obj_neural['AWA_nonlinear_params']['vm2']))
    winf = 0.5*(1 + np.tanh((v-params_obj_neural['AWA_nonlinear_params']['vm3'])/params_obj_neural['AWA_nonlinear_params']['vm4']))
    yinf = 0.5*(1 + np.tanh((v-params_obj_neural['AWA_nonlinear_params']['vb1'])/params_obj_neural['AWA_nonlinear_params']['vb2']))
    zinf = 0.5*(1 + np.tanh((v-params_obj_neural['AWA_nonlinear_params']['vs1'])/params_obj_neural['AWA_nonlinear_params']['vs2']))
    qinf = 0.5*(1 + np.tanh((v-params_obj_neural['AWA_nonlinear_params']['vq1'])/params_obj_neural['AWA_nonlinear_params']['vq2'])) 
    pinf = 0.5*(1 + np.tanh((v-params_obj_neural['AWA_nonlinear_params']['vp1'])/params_obj_neural['AWA_nonlinear_params']['vp2']))

    gkt = params_obj_neural['AWA_nonlinear_params']['TKL']+(params_obj_neural['AWA_nonlinear_params']['TKH']-params_obj_neural['AWA_nonlinear_params']['TKL'])*0.5*\
    (1+np.tanh(v-params_obj_neural['AWA_nonlinear_params']['vtk1'])/params_obj_neural['AWA_nonlinear_params']['vtk2'])

    tau = 1.0/np.cosh((v-params_obj_neural['AWA_nonlinear_params']['vt1'])/(2*params_obj_neural['AWA_nonlinear_params']['vt2']))
    kir = -np.log(1+np.exp(-0.2*(v-params_obj_neural['AWA_nonlinear_params']['vK']-params_obj_neural['AWA_nonlinear_params']['gKI'])))/0.2+params_obj_neural['AWA_nonlinear_params']['gKI']
    
    KCurr = np.zeros(params_obj_neural['N'])
    np.put(KCurr, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], (params_obj_neural['AWA_nonlinear_params']['gK']*w+params_obj_neural['AWA_nonlinear_params']['gK7']*\
        slo2+params_obj_neural['AWA_nonlinear_params']['gK4']*slo+params_obj_neural['AWA_nonlinear_params']['gK6']+params_obj_neural['AWA_nonlinear_params']['gK3']*\
        yinf*(1-bk)+params_obj_neural['AWA_nonlinear_params']['gK5']*kb)*(v-params_obj_neural['AWA_nonlinear_params']['vK']) + params_obj_neural['AWA_nonlinear_params']['gK2']*kir)

    """ calcium current """
    CaCurr = np.zeros(params_obj_neural['N'])
    np.put(CaCurr, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], params_obj_neural['AWA_nonlinear_params']['gCa']*(c1+params_obj_neural['AWA_nonlinear_params']['fac']*c2)*\
        (v-params_obj_neural['AWA_nonlinear_params']['vCa']))

    """ dw, etc """
    dv_AWA_vec = np.zeros(params_obj_neural['N'])
    dw_AWA_vec = np.zeros(params_obj_neural['N'])
    dc1_AWA_vec = np.zeros(params_obj_neural['N'])
    dc2_AWA_vec = np.zeros(params_obj_neural['N'])
    dbk_AWA_vec = np.zeros(params_obj_neural['N'])
    dslo_AWA_vec = np.zeros(params_obj_neural['N'])
    dslo2_AWA_vec = np.zeros(params_obj_neural['N'])
    dkb_AWA_vec = np.zeros(params_obj_neural['N']) 

    np.put(dv_AWA_vec, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], KCurr + CaCurr)
    np.put(dw_AWA_vec, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], (xinf-w)/params_obj_neural['AWA_nonlinear_params']['TK'])
    np.put(dc1_AWA_vec, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], (minf*winf/params_obj_neural['AWA_nonlinear_params']['mx']-c1)/params_obj_neural['AWA_nonlinear_params']['TC1']-
        minf*winf*c2/(params_obj_neural['AWA_nonlinear_params']['mx']*params_obj_neural['AWA_nonlinear_params']['TC1'])-c1/(2*params_obj_neural['AWA_nonlinear_params']['TC2']*tau)+c2/
        (2*params_obj_neural['AWA_nonlinear_params']['TC2']*tau))

    np.put(dc2_AWA_vec, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], (c1-c2)/(2*params_obj_neural['AWA_nonlinear_params']['TC2']*tau))
    np.put(dbk_AWA_vec, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], (yinf-bk)/params_obj_neural['AWA_nonlinear_params']['TbK'])
    np.put(dslo_AWA_vec, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], (zinf-slo)/params_obj_neural['AWA_nonlinear_params']['TS'])
    np.put(dslo2_AWA_vec, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], (qinf-slo2)/params_obj_neural['AWA_nonlinear_params']['TS2'])
    np.put(dkb_AWA_vec, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], (pinf-kb)/gkt)

    d_AWA_dyn_vec = np.concatenate((dv_AWA_vec, dw_AWA_vec, dc1_AWA_vec, dc2_AWA_vec, dbk_AWA_vec, dslo_AWA_vec, dslo2_AWA_vec, dkb_AWA_vec))

    return d_AWA_dyn_vec

def AVL_nonlinear_current(AVL_dyn_vec): 

    AVL_dict = params_obj_neural['AVL_nonlinear_params']

    v, C_1, C_2, C_3, O, m_UNC2, h_UNC2, m_EGL19, h_EGL19, m_CCA1, h_CCA1, m_SHL1, h_SHL1f, h_SHL1s, m_EGL36f, m_EGL36m, m_EGL36s = np.split(AVL_dyn_vec, 17)

    v = v[AVL_dict['AVL_ind']]
    C_1 = C_1[AVL_dict['AVL_ind']]
    C_2 = C_2[AVL_dict['AVL_ind']]
    C_3 = C_3[AVL_dict['AVL_ind']]
    O = O[AVL_dict['AVL_ind']]
    m_UNC2 = m_UNC2[AVL_dict['AVL_ind']]
    h_UNC2 = h_UNC2[AVL_dict['AVL_ind']]
    m_EGL19 = m_EGL19[AVL_dict['AVL_ind']]
    h_EGL19 = h_EGL19[AVL_dict['AVL_ind']]
    m_CCA1 = m_CCA1[AVL_dict['AVL_ind']]
    h_CCA1 = h_CCA1[AVL_dict['AVL_ind']]
    m_SHL1 = m_SHL1[AVL_dict['AVL_ind']]
    h_SHL1f = h_SHL1f[AVL_dict['AVL_ind']]
    h_SHL1s = h_SHL1s[AVL_dict['AVL_ind']]
    m_EGL36f = m_EGL36f[AVL_dict['AVL_ind']]
    m_EGL36m = m_EGL36m[AVL_dict['AVL_ind']]
    m_EGL36s = m_EGL36s[AVL_dict['AVL_ind']]

    # I_NCA
    I_NCA = AVL_dict['g_NCA'] * (v - AVL_dict['vNA'])

    # I_L
    I_L = AVL_dict['gL'] * (v - AVL_dict['vL'])

    # I_EXP_2
    I_EXP_2 = AVL_dict['g_EXP2'] * O * (v - AVL_dict['vK'])

    # I_UNC_2
    I_UNC_2 = AVL_dict['g_UNC2'] * m_UNC2**2 * h_UNC2 * (v - AVL_dict['vCA'])

    # I_EGL_19
    I_EGL_19 = AVL_dict['g_EGL19'] * m_EGL19 * h_EGL19 * (v - AVL_dict['vCA'])

    # I_CCA_1
    I_CCA_1 = AVL_dict['g_CCA1'] * m_CCA1**2 * h_CCA1 * (v - AVL_dict['vCA'])

    # I_SHL_1 
    I_SHL_1 = AVL_dict['g_SHL1'] * m_SHL1**3 * (0.7 * h_SHL1f + 0.3 * h_SHL1s) * (v - AVL_dict['vK'])

    # I_EGL_36
    I_EGL_36 = AVL_dict['g_EGL36'] * (0.33*m_EGL36f + 0.36*m_EGL36m + 0.39*m_EGL36s) * (v - AVL_dict['vK'])

    dV_AVL = I_NCA + I_L + I_EXP_2 + I_UNC_2 + I_EGL_19 + I_CCA_1 + I_SHL_1 + I_EGL_36

    # differentials for I_EXP_2
    a_1 = AVL_dict['a_1_p_1']*np.exp(AVL_dict['a_1_p_2']*v)
    a_2 = AVL_dict['a_2_p_7']*np.exp(AVL_dict['a_2_p_8']*v)
    a_i = AVL_dict['a_i_p_11']*np.exp(AVL_dict['a_i_p_12']*v)
    a_i2 = AVL_dict['a_i2_p_15']*np.exp(AVL_dict['a_i2_p_16']*v)

    b_1 = AVL_dict['b_1_p_3']*np.exp(-AVL_dict['b_1_p_4']*v)
    b_2 = AVL_dict['b_2_p_9']*np.exp(-AVL_dict['b_2_p_10']*v)
    b_i = AVL_dict['b_i_p_13']*np.exp(-AVL_dict['b_i_p_14']*v)

    psi = (b_2 * b_i * a_i2) / (a_2 * a_i)

    I = 1 - C_1 - C_2 - C_3 - O

    dC_1 = b_1 * C_2 - a_1 * C_1
    dC_2 = a_1 * C_1 + AVL_dict['K_b_p_6'] * C_3 - (b_1 + AVL_dict['K_f_p_5'])*C_2
    dC_3 = AVL_dict['K_f_p_5'] * C_2 + psi * I + b_2 * O - (AVL_dict['K_b_p_6'] + a_i2 + a_2) * C_3
    dO = b_i * I + a_2 * C_3 - (b_2 + a_i) * O

    # differentials for I_UNC_2
    a_m_UNC2 = (AVL_dict['a_m_UNC2_a'] * (v - AVL_dict['a_m_UNC2_b'])) / (1 - np.exp(-(v-AVL_dict['a_m_UNC2_b'])/AVL_dict['a_m_UNC2_c']))
    b_m_UNC2 = AVL_dict['b_m_UNC2_a'] * np.exp(-(v - AVL_dict['b_m_UNC2_b'])/AVL_dict['b_m_UNC2_c'])
    a_h_UNC2 = AVL_dict['a_h_UNC2_a'] * np.exp(-(v - AVL_dict['a_h_UNC2_b'])/AVL_dict['a_h_UNC2_c'])
    b_h_UNC2 = AVL_dict['b_h_UNC2_a'] / (1 + np.exp(-(v - AVL_dict['b_h_UNC2_b'])/AVL_dict['b_h_UNC2_c']))

    dm_UNC2 = a_m_UNC2 * (1 - m_UNC2) - b_m_UNC2 * m_UNC2
    dh_UNC2 = a_h_UNC2 * (1 - h_UNC2) - b_h_UNC2 * h_UNC2

    # differentials for I_EGL_19
    m_EGL19inf = 1 / (1 + np.exp(-(v-AVL_dict['m_EGL19inf_v_eq'])/AVL_dict['m_EGL19inf_k_a']))
    h_EGL19inf = (AVL_dict['h_EGL19inf_a'] / (1 + np.exp(-(v-AVL_dict['h_EGL19inf_v_eq_1'])/(AVL_dict['h_EGL19inf_k_i1']))) + AVL_dict['h_EGL19inf_b']) * (AVL_dict['h_EGL19inf_c'] / (1 + np.exp(-(v-AVL_dict['h_EGL19inf_v_eq_2'])/(AVL_dict['h_EGL19inf_k_i2']))) + AVL_dict['h_EGL19inf_d'])
    tau_m_EGL19 = (AVL_dict['tau_m_EGL19_a']* np.exp(-((v-AVL_dict['tau_m_EGL19_b'])/AVL_dict['tau_m_EGL19_c'])**2)) + (AVL_dict['tau_m_EGL19_d'] * np.exp(-((v-AVL_dict['tau_m_EGL19_e'])/AVL_dict['tau_m_EGL19_f'])**2)) + AVL_dict['tau_m_EGL19_g']
    tau_h_EGL19 = AVL_dict['tau_h_EGL19_a'] * (AVL_dict['tau_h_EGL19_b']/(1 + np.exp((v-AVL_dict['tau_h_EGL19_c'])/AVL_dict['tau_h_EGL19_d'])) + AVL_dict['tau_h_EGL19_e']/(1 + np.exp((v-AVL_dict['tau_h_EGL19_f'])/AVL_dict['tau_h_EGL19_g'])) + AVL_dict['tau_h_EGL19_h'])

    dm_EGL19 = (m_EGL19inf - m_EGL19) / tau_m_EGL19 
    dh_EGL19 = (h_EGL19inf - h_EGL19) / tau_h_EGL19 

    # differentials for I_CCA_1
    m_CCA1inf = 1 / (1 + np.exp(-(v-AVL_dict['m_CCA1inf_v_eq'])/AVL_dict['m_CCA1inf_k_a']))
    h_CCA1inf = 1 / (1 + np.exp((v-AVL_dict['h_CCA1inf_v_eq'])/AVL_dict['h_CCA1inf_k_i']))
    tau_m_CCA1 = AVL_dict['tau_m_CCA1_a'] / (1 + np.exp(-(v-AVL_dict['tau_m_CCA1_b'])/AVL_dict['tau_m_CCA1_c'])) + AVL_dict['tau_m_CCA1_d']
    tau_h_CCA1 = AVL_dict['tau_h_CCA1_a'] / (1 + np.exp((v-AVL_dict['tau_h_CCA1_b'])/AVL_dict['tau_h_CCA1_c'])) + AVL_dict['tau_h_CCA1_d']

    dm_CCA1 = (m_CCA1inf - m_CCA1) / tau_m_CCA1
    dh_CCA1 = (h_CCA1inf - h_CCA1) / tau_h_CCA1

    # differentials for I_SHL1
    m_SHL1inf = 1 / (1 + np.exp(-(v - AVL_dict['m_SHL1inf_v_eq'])/AVL_dict['m_SHL1inf_k_a']))
    h_SHL1inf = 1 / (1 + np.exp((v - AVL_dict['h_SHL1inf_v_eq'])/AVL_dict['h_SHL1inf_k_i']))
    tau_m_SHL1 = AVL_dict['tau_m_SHL1_a'] / (np.exp(-(v - AVL_dict['tau_m_SHL1_b'])/AVL_dict['tau_m_SHL1_c']) + np.exp((v - AVL_dict['tau_m_SHL1_d'])/AVL_dict['tau_m_SHL1_e'])) + AVL_dict['tau_m_SHL1_f'] 
    tau_h_SHL1f = AVL_dict['tau_h_SHL1f_a'] / (1 + np.exp((v - AVL_dict['tau_h_SHL1f_b'])/AVL_dict['tau_h_SHL1f_c'])) + AVL_dict['tau_h_SHL1f_d']
    tau_h_SHL1s = AVL_dict['tau_h_SHL1s_a'] / (1 + np.exp((v - AVL_dict['tau_h_SHL1s_b'])/AVL_dict['tau_h_SHL1s_c'])) + AVL_dict['tau_h_SHL1s_d']

    dm_SHL1 = (m_SHL1inf - m_SHL1) / tau_m_SHL1
    dh_SHL1f = (h_SHL1inf - h_SHL1f) / tau_h_SHL1f
    dh_SHL1s = (h_SHL1inf - h_SHL1s) / tau_h_SHL1s

    # differentials for I_EGL36
    m_EGL36inf = 1 / (1 + np.exp(-(v-AVL_dict['m_EGL36inf_v_eq'])/AVL_dict['m_EGL36inf_k_a']))

    dm_EGL36f = (m_EGL36inf - m_EGL36f) / AVL_dict['tau_m_EGL36f_tau_f']
    dm_EGL36m = (m_EGL36inf - m_EGL36m) / AVL_dict['tau_m_EGL36f_tau_m']
    dm_EGL36s = (m_EGL36inf - m_EGL36s) / AVL_dict['tau_m_EGL36f_tau_s']

    dv_AVL_vec = np.zeros(params_obj_neural['N'])
    dC_1_vec = np.zeros(params_obj_neural['N'])
    dC_2_vec = np.zeros(params_obj_neural['N'])
    dC_3_vec = np.zeros(params_obj_neural['N'])
    dO_vec = np.zeros(params_obj_neural['N'])
    dm_UNC2_vec = np.zeros(params_obj_neural['N'])
    dh_UNC2_vec = np.zeros(params_obj_neural['N'])
    dm_EGL19_vec = np.zeros(params_obj_neural['N'])
    dh_EGL19_vec = np.zeros(params_obj_neural['N'])
    dm_CCA1_vec = np.zeros(params_obj_neural['N'])
    dh_CCA1_vec = np.zeros(params_obj_neural['N'])
    dm_SHL1_vec = np.zeros(params_obj_neural['N'])
    dh_SHL1f_vec = np.zeros(params_obj_neural['N'])
    dh_SHL1s_vec = np.zeros(params_obj_neural['N'])
    dm_EGL36f_vec = np.zeros(params_obj_neural['N'])
    dm_EGL36m_vec = np.zeros(params_obj_neural['N'])
    dm_EGL36s_vec = np.zeros(params_obj_neural['N'])

    np.put(dv_AVL_vec, AVL_dict['AVL_ind'], dV_AVL)
    np.put(dC_1_vec, AVL_dict['AVL_ind'], dC_1)
    np.put(dC_2_vec, AVL_dict['AVL_ind'], dC_2)
    np.put(dC_3_vec, AVL_dict['AVL_ind'], dC_3)
    np.put(dO_vec, AVL_dict['AVL_ind'], dO)
    np.put(dm_UNC2_vec, AVL_dict['AVL_ind'], dm_UNC2)
    np.put(dh_UNC2_vec, AVL_dict['AVL_ind'], dh_UNC2)
    np.put(dm_EGL19_vec, AVL_dict['AVL_ind'], dm_EGL19)
    np.put(dh_EGL19_vec, AVL_dict['AVL_ind'], dh_EGL19)
    np.put(dm_CCA1_vec, AVL_dict['AVL_ind'], dm_CCA1)
    np.put(dh_CCA1_vec, AVL_dict['AVL_ind'], dh_CCA1)
    np.put(dm_SHL1_vec, AVL_dict['AVL_ind'], dm_SHL1)
    np.put(dh_SHL1f_vec, AVL_dict['AVL_ind'], dh_SHL1f)
    np.put(dh_SHL1s_vec, AVL_dict['AVL_ind'], dh_SHL1s)
    np.put(dm_EGL36f_vec, AVL_dict['AVL_ind'], dm_EGL36f)
    np.put(dm_EGL36m_vec, AVL_dict['AVL_ind'], dm_EGL36m)
    np.put(dm_EGL36s_vec, AVL_dict['AVL_ind'], dm_EGL36s)

    d_AVL_dyn_vec = np.concatenate([dv_AVL_vec, dC_1_vec, dC_2_vec, dC_3_vec, dO_vec, dm_UNC2_vec, dh_UNC2_vec, dm_EGL19_vec, dh_EGL19_vec, dm_CCA1_vec,
     dh_CCA1_vec, dm_SHL1_vec, dh_SHL1f_vec, dh_SHL1s_vec, dm_EGL36f_vec, dm_EGL36m_vec, dm_EGL36s_vec])

    return d_AVL_dyn_vec

def combine_baseline_nonlinear_currents(rhs_dict): # Needs to be simplified

    if params_obj_neural['nonlinear_AWA'] == True and params_obj_neural['nonlinear_AVL'] == False:

        AWA_dyn_vec = np.concatenate([rhs_dict['Vvec'], rhs_dict['y'][params_obj_neural['N'] * 2:]])
        dAWA_dyn_vec = AWA_nonlinear_current(AWA_dyn_vec)
        AWA_inds = params_obj_neural['AWA_nonlinear_params']['AWA_inds']

        dV = (-(rhs_dict['VsubEc'] + rhs_dict['GapCon'] + rhs_dict['SynapCon']) + rhs_dict['Input'])/params_obj_neural['C']

        np.put(rhs_dict['VsubEc'], AWA_inds, .25*10*(rhs_dict['Vvec'][AWA_inds] - (-65)))
        dV[AWA_inds] = (-(dAWA_dyn_vec[:params_obj_neural['N']][AWA_inds] + rhs_dict['VsubEc'][AWA_inds] + rhs_dict['GapCon'][AWA_inds] + rhs_dict['SynapCon'][AWA_inds]) + rhs_dict['Input'][AWA_inds])/params_obj_neural['C']
        dS = np.subtract(rhs_dict['SynRise'], rhs_dict['SynDrop'])

        return np.concatenate((dV, dS, dAWA_dyn_vec[params_obj_neural['N']:]))

    elif params_obj_neural['nonlinear_AWA'] == False and params_obj_neural['nonlinear_AVL'] == True:

        AVL_dyn_vec = np.concatenate([rhs_dict['Vvec'], rhs_dict['y'][params_obj_neural['N'] * 2:]])
        dAVL_dyn_vec = AVL_nonlinear_current(AVL_dyn_vec)
        AVL_ind = params_obj_neural['AVL_nonlinear_params']['AVL_ind']

        dV = (-(rhs_dict['VsubEc'] + rhs_dict['GapCon'] + rhs_dict['SynapCon']) + rhs_dict['Input'])/params_obj_neural['C']

        dV[AVL_ind] = (-(dAVL_dyn_vec[:params_obj_neural['N']][AVL_ind] + rhs_dict['GapCon'][AVL_ind] + rhs_dict['SynapCon'][AVL_ind]) + rhs_dict['Input'][AVL_ind])/params_obj_neural['AVL_nonlinear_params']['C_AVL']
        dS = np.subtract(rhs_dict['SynRise'], rhs_dict['SynDrop'])

        return np.concatenate((dV, dS, dAVL_dyn_vec[params_obj_neural['N']:]))

    else:

        AWA_dyn_vec = np.concatenate([rhs_dict['Vvec'], rhs_dict['y'][params_obj_neural['N'] * 2:9*params_obj_neural['N']]])
        AVL_dyn_vec = np.concatenate([rhs_dict['Vvec'], rhs_dict['y'][9*params_obj_neural['N']:]])

        dAWA_dyn_vec = AWA_nonlinear_current(AWA_dyn_vec)
        dAVL_dyn_vec = AVL_nonlinear_current(AVL_dyn_vec)
        AWA_inds = params_obj_neural['AWA_nonlinear_params']['AWA_inds']
        AVL_ind = params_obj_neural['AVL_nonlinear_params']['AVL_ind']

        dV = (-(rhs_dict['VsubEc'] + rhs_dict['GapCon'] + rhs_dict['SynapCon']) + rhs_dict['Input'])/params_obj_neural['C']

        np.put(rhs_dict['VsubEc'], AWA_inds, .25*10*(rhs_dict['Vvec'][AWA_inds] - (-65)))
        dV[AWA_inds] = (-(dAWA_dyn_vec[:params_obj_neural['N']][AWA_inds] + rhs_dict['VsubEc'][AWA_inds] + rhs_dict['GapCon'][AWA_inds] + rhs_dict['SynapCon'][AWA_inds]) + rhs_dict['Input'][AWA_inds])/params_obj_neural['C']
        dV[AVL_ind] = (-(dAVL_dyn_vec[:params_obj_neural['N']][AVL_ind] + rhs_dict['VsubEc'][AVL_ind] + rhs_dict['GapCon'][AVL_ind] + rhs_dict['SynapCon'][AVL_ind]) + rhs_dict['Input'][AVL_ind])/params_obj_neural['AVL_nonlinear_params']['C_AVL']
        dS = np.subtract(rhs_dict['SynRise'], rhs_dict['SynDrop'])

        return np.concatenate((dV, dS, dAWA_dyn_vec[params_obj_neural['N']:], dAVL_dyn_vec[params_obj_neural['N']:]))