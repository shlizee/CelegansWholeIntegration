
# coding: utf-8

########################################################################################################################################################################
# NETWORK SIMULATION MODULE ############################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

from julia.api import LibJulia
api = LibJulia.load()
api.init_julia(["check_bounds=yes"])

import time
import os

import numpy as np
import torch
import scipy.io as sio
import numba
from scipy import integrate, sparse, linalg, interpolate

from dynworm import sys_paths as paths
from dynworm import neural_params as n_params
from dynworm import body_params as b_params
from dynworm import utils
from diffeqpy import de, ode 

np.random.seed(10)

# TODO: REAL-TIME PLOTTING OF NEURAL VOLTAGE FOR SELECTED NEURON
# GENERALIZE PHYSIOLOGICAL PARAMETERS
# SUPPORT FOR NON_JULIA

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
            print('Nonlinear neural parameters enabled')

    else:

        assert type(custom_baseline_params) == dict, "Custom neural parameters should be of dictionary format"

        if validate_custom_neural_params(custom_baseline_params) == True:

            params_obj_neural = custom_baseline_params.copy()
            print('Accepted the custom neural parameters')

            if additional_params != False:

                params_obj_neural['nonlinear_AWA'] = additional_params['AWA']
                params_obj_neural['nonlinear_AVL'] = additional_params['AVL']
                print('Nonlinear neural parameters enabled')

def validate_custom_neural_params(custom_baseline_params):

    # TODO: Also check for dimensions

    key_checker = []

    for key in n_params.pA_unit_baseline.keys():
        
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

        params_obj_neural['ggap_total_mat'] = n_params.ggap_total_mat.copy()
        params_obj_neural['gsyn_max_mat'] = n_params.gsyn_max_mat.copy()

        params_obj_neural['Gg_Static'] = n_params.Gg_Static.copy()
        params_obj_neural['Gs_Static'] = n_params.Gs_Static.copy()

        EMat_mask = n_params.EMat_mask.copy()
        print('Using the default connectivity')

    else:

        assert type(custom_connectivity_dict) == dict, "Custom connectivity should be of dictionary format"

        params_obj_neural['ggap_total_mat'] = n_params.ggap_total_mat.copy()
        params_obj_neural['gsyn_max_mat'] = n_params.gsyn_max_mat.copy()

        params_obj_neural['Gg_Static'] = custom_connectivity_dict['gap'].copy()
        params_obj_neural['Gs_Static'] = custom_connectivity_dict['syn'].copy()

        EMat_mask = custom_connectivity_dict['directionality'].copy()
        print('Accepted the custom connectivity')

    params_obj_neural['EMat'] = params_obj_neural['E_rev'] * EMat_mask
    params_obj_neural['mask_Healthy'] = np.ones(params_obj_neural['N'], dtype = 'bool')

def query_params_obj_neural():

    return params_obj_neural

########################################################################################################################################################################
### MASTER EXECUTION FUNCTIONS (BASIC) #################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

def run_network_constinput(t_duration, input_vec, ablation_mask, \
    custom_initcond = False, ablation_type = "all", verbose = True, 
    use_julia_engine = False):

    assert 'params_obj_neural' in globals(), "Neural parameters and connectivity must be initialized before running the simulation"

    """ Set up simulation parameters """

    tf = t_duration * params_obj_neural['time_scaler']

    nsteps = int(np.floor(tf/params_obj_neural['dt']) + 1)
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

    print("Network integration prep completed...")

    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0 and use_julia_engine == False:

        r = integrate.ode(membrane_voltageRHS_constinput, compute_jacobian_constinput).set_integrator('vode', rtol = 1e-8, atol = 1e-8, method = 'bdf')
        r.set_initial_value(initcond, 0)

        """ Additional Python step to store the trajectories """
        t = np.zeros(nsteps)
        traj = np.zeros((nsteps, params_obj_neural['N']))

        t[0] = 0
        traj[0, :] = initcond[:params_obj_neural['N']]
        vthmat = np.tile(params_obj_neural['vth'], (nsteps, 1))

        """ Integrate the ODE(s) across each delta_t timestep """

        print("Computing network dynamics...")

        k = 1

        while r.successful() and k < nsteps:

            r.integrate(r.t + params_obj_neural['dt'])

            t[k] = r.t
            traj[k, :] = r.y[:params_obj_neural['N']]

            k += 1

            if verbose == True:

                if k in progress_milestones:

                    print(str(np.round((float(k) / nsteps) * 100, 0)) + '% ' + 'completed')

    else:

        print("Computing network dynamics with Julia engine...")

        r = de.ODEProblem(membrane_voltageRHS_constinput_julia, initcond, (0, tf))
        sol = de.solve(r, de.QNDF(autodiff=False), saveat = params_obj_neural['dt'], reltol = 1e-8, abstol = 1e-8, save_everystep=False)

        vthmat = np.tile(params_obj_neural['vth'], (nsteps, 1))

        t = sol.t
        traj = np.vstack(sol.u)[:, :params_obj_neural['N']]

    result_dict_network = {
            "t": t,
            "steps": nsteps,
            "raw_v_solution": traj,
            "v_threshold": vthmat,
            "v_solution" : voltage_filter(np.subtract(traj, vthmat), 200, 1)
            }

    return result_dict_network

def run_network_dyninput(input_mat, ablation_mask, \
    custom_initcond = False, ablation_type = "all", \
    interp_kind_input = 'nearest', verbose = True, 
    use_julia_engine = False):

    assert 'params_obj_neural' in globals(), "Neural parameters and connectivity must be initialized before running the simulation"

    """ define pre-computed noise input matrix """

    t0 = 0
    tf = (len(input_mat) - 1) * params_obj_neural['dt'] # the last timepoint to be computed

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

    print("Network integration prep completed...")

    """ Configuring the ODE Solver """

    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0 and use_julia_engine == False:

        r = integrate.ode(membrane_voltageRHS_dyninput, compute_jacobian_dyninput).set_integrator('vode', rtol = 1e-8, atol = 1e-8, method = 'bdf')
        r.set_initial_value(initcond, t0)

        """ Additional Python step to store the trajectories """
        t = np.zeros(nsteps)
        traj = np.zeros((nsteps, params_obj_neural['N']))
        vthmat = np.zeros((nsteps, params_obj_neural['N']))

        t[0] = t0
        traj[0, :] = initcond[:params_obj_neural['N']]
        vthmat[0, :] = EffVth_rhs(input_mat[0, :])

        """ Integrate the ODE(s) across each delta_t timestep """

        print("Computing network dynamics...")

        k = 1

        while r.successful() and k < nsteps:

            r.integrate(r.t + params_obj_neural['dt'])

            t[k] = r.t
            traj[k, :] = r.y[:params_obj_neural['N']]
            vthmat[k, :] = EffVth_rhs(input_mat[k, :])

            k += 1

            if verbose == True:

                if k in progress_milestones:

                    print(str(np.round((float(k) / nsteps) * 100, 0)) + '% ' + 'completed')

    else:

        print("Computing network dynamics with Julia engine...")

        r = de.ODEProblem(membrane_voltageRHS_dyninput_julia, initcond, (0, tf))
        sol = de.solve(r, de.QNDF(autodiff=False), saveat = params_obj_neural['dt'], reltol = 1e-8, abstol = 1e-8, save_everystep=False)

        vthmat = np.zeros((nsteps, params_obj_neural['N']))

        for k in range(len(input_mat)):

            vthmat[k, :] = EffVth_rhs(input_mat[k, :])

        t = sol.t
        traj = np.vstack(sol.u)[:, :params_obj_neural['N']]

    result_dict_network = {
            "t": t,
            "steps": nsteps,
            "raw_v_solution": traj,
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
    custom_initcond = False, ablation_type = "all", \
    interp_kind_input = 'nearest', interp_kind_voltage = 'linear', verbose = True, 
    use_julia_engine = False):
    
    assert 'params_obj_neural' in globals(), "Neural parameters and connectivity must be initialized before running the simulation"
    assert len(input_mat) == len(ext_voltage_mat), "Length of input_mat and ext_voltage_mat should be identical"
    # Also Check dimensions of input_mat and ext_voltage_mat

    t0 = 0
    tf = (len(input_mat) - 1) * params_obj_neural['dt']

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

    print("Network integration prep completed...")

    """ Configuring the ODE Solver """

    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0 and use_julia_engine == False:

        r = integrate.ode(membrane_voltageRHS_vext, compute_jacobian_vext).set_integrator('vode', rtol = 1e-8, atol = 1e-8, method = 'bdf')
        r.set_initial_value(initcond, t0)

        """ Additional Python step to store the trajectories """
        t = np.zeros(nsteps)
        traj = np.zeros((nsteps, params_obj_neural['N']))
        vthmat = np.zeros((nsteps, params_obj_neural['N']))

        t[0] = t0
        traj[0, :] = initcond[:params_obj_neural['N']]
        vthmat[0, :] = EffVth_rhs(input_mat[0, :])

        """ Integrate the ODE(s) across each delta_t timestep """

        print("Computing network dynamics...")

        k = 1

        while r.successful() and k < nsteps:

            r.integrate(r.t + params_obj_neural['dt'])

            t[k] = r.t
            traj[k, :] = r.y[:params_obj_neural['N']]
            vthmat[k, :] = EffVth_rhs(input_mat[k, :])

            k += 1

            if verbose == True:

                if k in progress_milestones:

                    print(str(np.round((float(k) / nsteps) * 100, 0)) + '% ' + 'completed')

    else:

        print("Computing network dynamics with Julia engine...")

        r = de.ODEProblem(membrane_voltageRHS_vext_julia, initcond, (0, tf))
        sol = de.solve(r, de.QNDF(autodiff=False), saveat = params_obj_neural['dt'], reltol = 1e-8, abstol = 1e-8, save_everystep=False)

        vthmat = np.zeros((nsteps, params_obj_neural['N']))

        for k in range(len(input_mat)):

            vthmat[k, :] = EffVth_rhs(input_mat[k, :])

        t = sol.t
        traj = np.vstack(sol.u)[:, :params_obj_neural['N']]

    result_dict_network = {
            "t": t,
            "steps": nsteps,
            "raw_v_solution": traj,
            "v_threshold": vthmat,
            "v_solution" : voltage_filter(np.subtract(traj, vthmat), 200, 1)
            }

    return result_dict_network

def run_network_fdb(input_mat, ablation_mask, \
    custom_initcond = False, ablation_type = "all", \
    interp_kind_input = 'nearest', \
    custom_muscle_map = False, fdb_init = 1.38, t_delay = 0.54, reaction_scaling = 1, verbose = True, 
    use_julia_engine = False):

    assert 'params_obj_neural' in globals(), "Neural parameters and connectivity must be initialized before running the simulation"

    t0 = 0
    tf = (len(input_mat) - 1) * params_obj_neural['dt']

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

    """ Additional Python step to store the trajectories """

    t = np.zeros(nsteps)
    traj = np.zeros((nsteps, params_obj_neural['N']))
    vthmat = np.zeros((nsteps, params_obj_neural['N']))

    t[0] = t0
    traj[0, :] = initcond[:params_obj_neural['N']]
    vthmat[0, :] = EffVth_rhs(input_mat[0, :])

    params_obj_neural['interpolate_traj'] = interpolate.interp1d(timepoints, traj, axis=0, fill_value = "extrapolate")
    params_obj_neural['interpolate_vthmat'] = interpolate.interp1d(timepoints, vthmat, axis=0, fill_value = "extrapolate")

    print("Network integration prep completed...")

    """ Configuring the ODE Solver """

    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0 and use_julia_engine == False:

        r = integrate.ode(membrane_voltageRHS_fdb, compute_jacobian_fdb).set_integrator('vode', rtol = 1e-8, atol = 1e-8, method = 'bdf')
        r.set_initial_value(initcond, t0)

        """ Integrate the ODE(s) across each delta_t timestep """

        print("Computing network dynamics...")

        k = 1

        while r.successful() and k < nsteps:

            r.integrate(r.t + params_obj_neural['dt'])

            t[k] = r.t
            traj[k, :] = r.y[:params_obj_neural['N']]
            vthmat[k, :] = EffVth_rhs(input_mat[k, :])

            params_obj_neural['interpolate_traj'] = interpolate.interp1d(timepoints, traj, axis=0, fill_value = "extrapolate")
            params_obj_neural['interpolate_vthmat'] = interpolate.interp1d(timepoints, vthmat, axis=0, fill_value = "extrapolate")

            k += 1

            if verbose == True:

                if k in progress_milestones:

                    print(str(np.round((float(k) / nsteps) * 100, 0)) + '% ' + 'completed')

    else:

        tspan = (t0, tf)
        r = de.ODEProblem(membrane_voltageRHS_fdb_julia, initcond, tspan)
        integrator = de.init(r, de.KenCarp47(autodiff=False), reltol = 1e-8, abstol = 1e-8, save_everystep=False)
        timepoints_ = timepoints[:-1]

        print("Computing network dynamics with Julia engine...")

        k = 1

        for integrator_sol in de.TimeChoiceIterator(integrator, timepoints_):

            t[k] = integrator_sol[1] + params_obj_neural['dt']
            traj[k, :] = integrator_sol[0][:params_obj_neural['N']]
            vthmat[k, :] = EffVth_rhs(input_mat[k, :])

            params_obj_neural['interpolate_traj'] = interpolate.interp1d(timepoints, traj, axis=0, fill_value = "extrapolate")
            params_obj_neural['interpolate_vthmat'] = interpolate.interp1d(timepoints, vthmat, axis=0, fill_value = "extrapolate")

            k += 1

            if verbose == True:

                if k in progress_milestones:

                    print(str(np.round((float(k) / nsteps) * 100, 0)) + '% ' + 'completed')

    result_dict_network = {
            "t": t,
            "steps": nsteps,
            "raw_v_solution": traj,
            "v_threshold": vthmat,
            "v_solution" : voltage_filter(np.subtract(traj, vthmat), 200, 1)
            }

    return result_dict_network

def run_network_fdb_externalV(input_mat, ext_voltage_mat, ablation_mask, \
    custom_initcond = False, ablation_type = "all", \
    interp_kind_input = 'nearest', interp_kind_voltage = 'linear', \
    custom_muscle_map = False, fdb_init = 1.38, t_delay = 0.54, reaction_scaling = 1, verbose = True, 
    use_julia_engine = False):

    assert 'params_obj_neural' in globals(), "Neural parameters and connectivity must be initialized before running the simulation"
    assert len(input_mat) == len(ext_voltage_mat), "Length of input_mat and ext_voltage_mat should be identical"

    t0 = 0
    tf = (len(input_mat) - 1) * params_obj_neural['dt']

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

    """ Additional Python step to store the trajectories """

    t = np.zeros(nsteps)
    traj = np.zeros((nsteps, params_obj_neural['N']))
    vthmat = np.zeros((nsteps, params_obj_neural['N']))

    t[0] = t0
    traj[0, :] = initcond[:params_obj_neural['N']]
    vthmat[0, :] = EffVth_rhs(input_mat[0, :])

    params_obj_neural['interpolate_traj'] = interpolate.interp1d(timepoints, traj, axis=0, fill_value = "extrapolate")
    params_obj_neural['interpolate_vthmat'] = interpolate.interp1d(timepoints, vthmat, axis=0, fill_value = "extrapolate")

    print("Network integration prep completed...")

    """ Configuring the ODE Solver """

    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0 and use_julia_engine == False:

        r = integrate.ode(membrane_voltageRHS_fdb, compute_jacobian_fdb).set_integrator('vode', rtol = 1e-8, atol = 1e-8, method = 'bdf')
        r.set_initial_value(initcond, t0)

        """ Integrate the ODE(s) across each delta_t timestep """

        print("Computing network dynamics...")

        k = 1

        while r.successful() and k < nsteps:

            r.integrate(r.t + params_obj_neural['dt'])

            t[k] = r.t
            traj[k, :] = r.y[:params_obj_neural['N']]
            vthmat[k, :] = EffVth_rhs(input_mat[k, :])

            params_obj_neural['interpolate_traj'] = interpolate.interp1d(timepoints, traj, axis=0, fill_value = "extrapolate")
            params_obj_neural['interpolate_vthmat'] = interpolate.interp1d(timepoints, vthmat, axis=0, fill_value = "extrapolate")

            k += 1

            if verbose == True:

                if k in progress_milestones:

                    print(str(np.round((float(k) / nsteps) * 100, 0)) + '% ' + 'completed')

    else:

        tspan = (t0, tf)
        r = de.ODEProblem(membrane_voltageRHS_fdb_vext_julia, initcond, tspan)
        integrator = de.init(r, de.KenCarp47(autodiff=False), reltol = 1e-8, abstol = 1e-8, save_everystep=False)
        timepoints_ = timepoints[:-1]

        print("Computing network dynamics with Julia engine...")

        k = 1

        for integrator_sol in de.TimeChoiceIterator(integrator, timepoints_):

            t[k] = integrator_sol[1] + params_obj_neural['dt']
            traj[k, :] = integrator_sol[0][:params_obj_neural['N']]
            vthmat[k, :] = EffVth_rhs(input_mat[k, :])

            params_obj_neural['interpolate_traj'] = interpolate.interp1d(timepoints, traj, axis=0, fill_value = "extrapolate")
            params_obj_neural['interpolate_vthmat'] = interpolate.interp1d(timepoints, vthmat, axis=0, fill_value = "extrapolate")

            k += 1

            if verbose == True:

                if k in progress_milestones:

                    print(str(np.round((float(k) / nsteps) * 100, 0)) + '% ' + 'completed')

    result_dict_network = {
            "t": t,
            "steps": nsteps,
            "raw_v_solution": traj,
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
        np.put(voltage_initcond, params_obj_neural['AVL_nonlinear_params']['AVL_ind'], -60)
        synaptic_initcond = 10**(-4)*np.random.normal(0, 0.94, params_obj_neural['N'])
        avl_initcond = np.zeros(params_obj_neural['N'] * 16,)
        avl_initcond[12 * 279 + params_obj_neural['AVL_nonlinear_params']['AVL_ind']] = 1

        full_initcond = np.concatenate([voltage_initcond, synaptic_initcond, avl_initcond])

    # AWA + AVL nonlinear

    else:

        params_obj_neural['AWA_nonlinear_params'] = n_params.AWA_nonlinear_params.copy()
        params_obj_neural['AVL_nonlinear_params'] = n_params.AVL_nonlinear_params.copy()

        voltage_initcond = 10**(-4)*np.random.normal(0, 0.94, params_obj_neural['N'])
        np.put(voltage_initcond, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], -74.99)
        np.put(voltage_initcond, params_obj_neural['AVL_nonlinear_params']['AVL_ind'], -60)
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
        avl_initcond[12 * 279 + params_obj_neural['AVL_nonlinear_params']['AVL_ind']] = 1

        full_initcond = np.concatenate([voltage_initcond, synaptic_initcond, awa_initcond1, awa_initcond2, awa_initcond3, awa_initcond4,
         awa_initcond5, awa_initcond6, awa_initcond7, avl_initcond])

    return full_initcond

def EffVth(Gg, Gs):

    Gcmat = np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    EcVec = np.multiply(params_obj_neural['Ec'], np.ones((params_obj_neural['N'], 1)))

    M1 = -Gcmat
    b1 = np.multiply(params_obj_neural['Gc'], EcVec)

    Ggap = np.multiply(params_obj_neural['ggap_total_mat'], Gg)
    Ggapdiag = np.subtract(Ggap, np.diag(np.diag(Ggap)))
    Ggapsum = Ggapdiag.sum(axis = 1)
    Ggapsummat = sparse.spdiags(Ggapsum, 0, params_obj_neural['N'], params_obj_neural['N']).toarray()
    M2 = -np.subtract(Ggapsummat, Ggapdiag)

    Gs_ij = np.multiply(params_obj_neural['gsyn_max_mat'], Gs)
    s_eq = round((params_obj_neural['ar']/(params_obj_neural['ar'] + 2 * params_obj_neural['ad'])), 4)
    sjmat = np.multiply(s_eq, np.ones((params_obj_neural['N'], params_obj_neural['N'])))
    S_eq = np.multiply(s_eq, np.ones((params_obj_neural['N'], 1)))
    Gsyn = np.multiply(sjmat, Gs_ij)
    Gsyndiag = np.subtract(Gsyn, np.diag(np.diag(Gsyn)))
    Gsynsum = Gsyndiag.sum(axis = 1)
    M3 = -sparse.spdiags(Gsynsum, 0, params_obj_neural['N'], params_obj_neural['N']).toarray()

    #b3 = np.dot(Gs_ij, np.multiply(s_eq, params_obj_neural['E']))
    b3 = np.dot(np.multiply(Gs_ij, params_obj_neural['EMat']), S_eq)

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
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, params_obj_neural['vth'])))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], params_obj_neural['inmask'])

    """ dV and dS and merge them back to dydt """

    dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def membrane_voltageRHS_constinput_julia(dy, y, p, t):

    Vvec, SVec = np.split(y[:params_obj_neural['N']*2], 2)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, params_obj_neural['vth'])))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], params_obj_neural['inmask'])

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

    dy[:] = d_network

def compute_jacobian_constinput(t, y):

    Vvec, SVec = np.split(y, 2)
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap_total_mat'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn_max_mat'], params_obj_neural['Gs_Dynamic'])
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
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

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

    dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def membrane_voltageRHS_dyninput_julia(dy, y, p, t):

    Vvec, SVec = np.split(y[:params_obj_neural['N']*2], 2)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

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

    dy[:] = d_network

def compute_jacobian_dyninput(t, y):

    Vvec, SVec = np.split(y, 2)
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap_total_mat'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn_max_mat'], params_obj_neural['Gs_Dynamic'])
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
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

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

    dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def membrane_voltageRHS_vext_julia(dy, y, p, t):

    Vvec, SVec = np.split(y[:params_obj_neural['N']*2], 2)

    v_ext = params_obj_neural['interpolate_voltage'](t)
    Vvec = np.add(Vvec, v_ext)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

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

    dy[:] = d_network

def compute_jacobian_vext(t, y):

    Vvec, SVec = np.split(y, 2)
    v_ext = params_obj_neural['interpolate_voltage'](t)
    Vvec = np.add(Vvec, v_ext)
    
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap_total_mat'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn_max_mat'], params_obj_neural['Gs_Dynamic'])
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
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

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
    dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def membrane_voltageRHS_fdb_julia(dy, y, p, t):

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
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

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

    dy[:] = d_network

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
    Ggap = np.multiply(params_obj_neural['ggap_total_mat'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn_max_mat'], params_obj_neural['Gs_Dynamic'])
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
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

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
    dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def membrane_voltageRHS_fdb_vext_julia(dy, y, p, t):

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
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

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

    dy[:] = d_network

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
    Ggap = np.multiply(params_obj_neural['ggap_total_mat'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn_max_mat'], params_obj_neural['Gs_Dynamic'])
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
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ interpolate vth """
    vth = EffVth_rhs(params_obj_neural['inmask'])

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth)))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], params_obj_neural['inmask'])

    """ dV and dS and merge them back to dydt """
    dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def membrane_voltageRHS_fdb_vext_spatialgrad_julia(dy, y, p, t):

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
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ interpolate vth """
    vth = EffVth_rhs(params_obj_neural['inmask'])

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth)))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], params_obj_neural['inmask'])

    """ dV and dS and merge them back to dydt """
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

    dy[:] = d_network

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
    Ggap = np.multiply(params_obj_neural['ggap_total_mat'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn_max_mat'], params_obj_neural['Gs_Dynamic'])
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
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ refer input """
    vth = EffVth_rhs(params_obj_neural['inmask'])

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth)))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], params_obj_neural['inmask'])

    """ dV and dS and merge them back to dydt """
    dV = (-(VsubEc + GapCon + SynapCon) + Input)/params_obj_neural['C']
    dS = np.subtract(SynRise, SynDrop)

    return np.concatenate((dV, dS))

def membrane_voltageRHS_vext_spatialgrad_julia(dy, y, p, t):

    Vvec, SVec = np.split(y[:params_obj_neural['N']*2], 2)

    v_ext = params_obj_neural['interpolate_voltage'](t)
    Vvec = np.add(Vvec, v_ext)

    """ Gc(Vi - Ec) """
    VsubEc = np.multiply(params_obj_neural['Gc'], (Vvec - params_obj_neural['Ec']))

    """ Gg(Vi - Vj) computation """
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))
    GapCon = np.multiply(np.multiply(params_obj_neural['Gg_Dynamic'], params_obj_neural['ggap_total_mat']), np.subtract(np.transpose(Vrep), Vrep)).sum(axis = 1)

    """ Gs*S*(Vi - Ej) Computation """
    VsubEj = np.subtract(np.transpose(Vrep), params_obj_neural['EMat'])
    SynapCon = np.multiply(np.multiply(np.multiply(params_obj_neural['Gs_Dynamic'], params_obj_neural['gsyn_max_mat']), np.tile(SVec, (params_obj_neural['N'], 1))), VsubEj).sum(axis = 1)

    """ refer input """
    vth = EffVth_rhs(params_obj_neural['inmask'])

    """ ar*(1-Si)*Sigmoid Computation """
    SynRise = np.multiply(np.multiply(params_obj_neural['ar'], (np.subtract(1.0, SVec))),
                          np.reciprocal(1.0 + np.exp(-params_obj_neural['B']*(np.subtract(Vvec, vth)))))

    SynDrop = np.multiply(params_obj_neural['ad'], SVec)

    """ Input Mask """
    Input = np.multiply(params_obj_neural['iext'], params_obj_neural['inmask'])

    """ dV and dS and merge them back to dydt """
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

    dy[:] = d_network

def compute_jacobian_vext_spatialgrad(t, y):

    Vvec, SVec = np.split(y, 2)

    v_ext = params_obj_neural['interpolate_voltage'](t)
    Vvec = np.add(Vvec, v_ext)
    
    Vrep = np.tile(Vvec, (params_obj_neural['N'], 1))

    J1_M1 = -np.multiply(params_obj_neural['Gc'], np.eye(params_obj_neural['N']))
    Ggap = np.multiply(params_obj_neural['ggap_total_mat'], params_obj_neural['Gg_Dynamic'])
    Ggapsumdiag = -np.diag(Ggap.sum(axis = 1))
    J1_M2 = np.add(Ggap, Ggapsumdiag) 
    Gsyn = np.multiply(params_obj_neural['gsyn_max_mat'], params_obj_neural['Gs_Dynamic'])
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

#########################################################################################################################################################################
### NON LINEAR CURRENTS #################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

def AWA_nonlinear_current(AWA_dyn_vec):

    AWA_dict = params_obj_neural['AWA_nonlinear_params']

    Vvec, wVec, c1Vec, c2Vec, bkVec, sloVec, slo2Vec, kbVec = np.split(AWA_dyn_vec, 8)

    """ channel variables """

    v = Vvec[AWA_dict['AWA_inds']]
    w = wVec[AWA_dict['AWA_inds']]
    c1 = c1Vec[AWA_dict['AWA_inds']]
    c2 = c2Vec[AWA_dict['AWA_inds']]
    bk = bkVec[AWA_dict['AWA_inds']]
    slo = sloVec[AWA_dict['AWA_inds']]
    slo2 = slo2Vec[AWA_dict['AWA_inds']]
    kb = kbVec[AWA_dict['AWA_inds']]

    """ leak current """

    LCurr = AWA_dict['gL']*(v - AWA_dict['vL'])

    """ potassium current """

    xinf = 0.5*(1 + np.tanh((v-AWA_dict['vk1'])/AWA_dict['vk2']))
    minf = 0.5*(1 + np.tanh((v-AWA_dict['vm1'])/AWA_dict['vm2']))
    winf = 0.5*(1 + np.tanh((v-AWA_dict['vm3'])/AWA_dict['vm4']))
    yinf = 0.5*(1 + np.tanh((v-AWA_dict['vb1'])/AWA_dict['vb2']))
    zinf = 0.5*(1 + np.tanh((v-AWA_dict['vs1'])/AWA_dict['vs2']))
    qinf = 0.5*(1 + np.tanh((v-AWA_dict['vq1'])/AWA_dict['vq2'])) 
    pinf = 0.5*(1 + np.tanh((v-AWA_dict['vp1'])/AWA_dict['vp2']))

    gkt = AWA_dict['TKL']+(AWA_dict['TKH']-AWA_dict['TKL'])*0.5*(1+np.tanh(v-AWA_dict['vtk1'])/AWA_dict['vtk2'])

    tau = 1.0/np.cosh((v-AWA_dict['vt1'])/(2*AWA_dict['vt2']))
    kir = -np.log(1+np.exp(-0.2*(v-AWA_dict['vK']-AWA_dict['gKI'])))/0.2+AWA_dict['gKI']
    
    KCurr = (AWA_dict['gK']*w+AWA_dict['gK7']*slo2+AWA_dict['gK4']*slo+AWA_dict['gK6']+AWA_dict['gK3']*yinf*(1-bk)+AWA_dict['gK5']*kb)*(v-AWA_dict['vK']) + AWA_dict['gK2']*kir

    """ calcium current """

    CaCurr = AWA_dict['gCa']*(c1+AWA_dict['fac']*c2)*(v-AWA_dict['vCa'])

    dv_AWA_vec = np.zeros(params_obj_neural['N'])
    dw_AWA_vec = np.zeros(params_obj_neural['N'])
    dc1_AWA_vec = np.zeros(params_obj_neural['N'])
    dc2_AWA_vec = np.zeros(params_obj_neural['N'])
    dbk_AWA_vec = np.zeros(params_obj_neural['N'])
    dslo_AWA_vec = np.zeros(params_obj_neural['N'])
    dslo2_AWA_vec = np.zeros(params_obj_neural['N'])
    dkb_AWA_vec = np.zeros(params_obj_neural['N']) 

    np.put(dv_AWA_vec, AWA_dict['AWA_inds'], LCurr + KCurr + CaCurr)
    np.put(dw_AWA_vec, AWA_dict['AWA_inds'], (xinf-w)/AWA_dict['TK'])
    np.put(dc1_AWA_vec, AWA_dict['AWA_inds'], (minf*winf/AWA_dict['mx']-c1)/AWA_dict['TC1']-minf*winf*c2/(AWA_dict['mx']*AWA_dict['TC1'])-c1/(2*AWA_dict['TC2']*tau)+c2/(2*AWA_dict['TC2']*tau))
    np.put(dc2_AWA_vec, AWA_dict['AWA_inds'], (c1-c2)/(2*AWA_dict['TC2']*tau))
    np.put(dbk_AWA_vec, AWA_dict['AWA_inds'], (yinf-bk)/AWA_dict['TbK'])
    np.put(dslo_AWA_vec, AWA_dict['AWA_inds'], (zinf-slo)/AWA_dict['TS'])
    np.put(dslo2_AWA_vec, AWA_dict['AWA_inds'], (qinf-slo2)/AWA_dict['TS2'])
    np.put(dkb_AWA_vec, AWA_dict['AWA_inds'], (pinf-kb)/gkt)

    d_AWA_dyn_vec = np.concatenate((dv_AWA_vec, dw_AWA_vec, dc1_AWA_vec, dc2_AWA_vec, dbk_AWA_vec, dslo_AWA_vec, dslo2_AWA_vec, dkb_AWA_vec))

    return d_AWA_dyn_vec

def AVL_nonlinear_current(AVL_dyn_vec): 

    AVL_dict = params_obj_neural['AVL_nonlinear_params']

    V, C1_EXP2, C2_EXP2, C3_EXP2, O_EXP2, m_UNC2, h_UNC2, m_EGL19, h_EGL19, m_CCA1, h_CCA1, m_SHL1, hf_SHL1, hs_SHL1, mf_EGL36, mm_EGL36, ms_EGL36 = np.split(AVL_dyn_vec, 17)

    V = V[AVL_dict['AVL_ind']]
    C1_EXP2 = C1_EXP2[AVL_dict['AVL_ind']]
    C2_EXP2 = C2_EXP2[AVL_dict['AVL_ind']]
    C3_EXP2 = C3_EXP2[AVL_dict['AVL_ind']]
    O_EXP2 = O_EXP2[AVL_dict['AVL_ind']]
    m_UNC2 = m_UNC2[AVL_dict['AVL_ind']]
    h_UNC2 = h_UNC2[AVL_dict['AVL_ind']]
    m_EGL19 = m_EGL19[AVL_dict['AVL_ind']]
    h_EGL19 = h_EGL19[AVL_dict['AVL_ind']]
    m_CCA1 = m_CCA1[AVL_dict['AVL_ind']]
    h_CCA1 = h_CCA1[AVL_dict['AVL_ind']]
    m_SHL1 = m_SHL1[AVL_dict['AVL_ind']]
    hf_SHL1 = hf_SHL1[AVL_dict['AVL_ind']]
    hs_SHL1 = hs_SHL1[AVL_dict['AVL_ind']]
    mf_EGL36 = mf_EGL36[AVL_dict['AVL_ind']]
    mm_EGL36 = mm_EGL36[AVL_dict['AVL_ind']]
    ms_EGL36 = ms_EGL36[AVL_dict['AVL_ind']]
    I_EXP2_ = 1 - C1_EXP2 - C2_EXP2 - C3_EXP2 - O_EXP2

    I_UNC2 = AVL_dict['g_UNC2'] * m_UNC2**2 * h_UNC2 * (V - AVL_dict['v_Ca'])
    I_EGL19 = AVL_dict['g_EGL19'] * m_EGL19 * h_EGL19 * (V - AVL_dict['v_Ca'])
    I_CCA1 = AVL_dict['g_CCA1'] * m_CCA1**2 * h_CCA1 * (V - AVL_dict['v_Ca'])
    I_SHL1 = AVL_dict['g_SHL1'] * m_SHL1**3 * (0.7 * hf_SHL1 + 0.3 * hs_SHL1) * (V - AVL_dict['v_K'])
    I_EGL36 = AVL_dict['g_EGL36'] * (0.31*mf_EGL36 + 0.36*mm_EGL36 + 0.39*ms_EGL36) * (V - AVL_dict['v_K'])
    I_EXP2 = AVL_dict['g_EXP2'] * O_EXP2 * (V - AVL_dict['v_K'])
    I_NCA = AVL_dict['g_NCA'] * (V - AVL_dict['v_Na'])
    I_L = AVL_dict['g_L'] * (V - AVL_dict['v_L'])

    dV_AVL = I_UNC2 + I_EGL19 + I_CCA1 + I_SHL1 + I_EGL36 + I_EXP2 + I_NCA + I_L   

    # differentials for I_UNC_2

    m_alpha = AVL_dict['m_a'] * (V-AVL_dict['m_b']) / (1 - np.exp(-(V-AVL_dict['m_b'])/AVL_dict['m_c']))
    m_beta  = AVL_dict['m_d'] * np.exp(-(V-AVL_dict['m_e'])/AVL_dict['m_f'])
    h_alpha = AVL_dict['h_a'] * np.exp(-(V-AVL_dict['h_b'])/AVL_dict['h_c'])
    h_beta  = AVL_dict['h_d'] / (1 + np.exp(-(V-AVL_dict['h_e'])/AVL_dict['h_f']))
    dmUNC2dt = m_alpha * (1 - m_UNC2) - m_beta * m_UNC2
    dhUNC2dt = h_alpha * (1 - h_UNC2) - h_beta * h_UNC2

    # differentials for I_EGL_19

    tau_m_EGL19 = AVL_dict['s_1'] * np.exp(-((V-AVL_dict['s_2'])/AVL_dict['s_3'])**2) + AVL_dict['s_4'] * np.exp(-((V-AVL_dict['s_5'])/AVL_dict['s_6'])**2) + AVL_dict['s_7']
    tau_h_EGL19 = AVL_dict['s_8'] * (AVL_dict['s_9'] / (1 + np.exp((V-AVL_dict['s_10'])/AVL_dict['s_11'])) + AVL_dict['s_12'] / (1 + np.exp((V-AVL_dict['s_13'])/AVL_dict['s_14'])) + AVL_dict['s_15'])
    m_EGL19_inf = 1 / (1 + np.exp(-(V-AVL_dict['q_1'])/AVL_dict['q_2']))
    h_EGL19_inf = (AVL_dict['q_3'] / (1 + np.exp(-(V-AVL_dict['q_4'])/AVL_dict['q_5'])) + AVL_dict['q_6']) * (AVL_dict['q_7'] / (1 + np.exp((V-AVL_dict['q_8'])/AVL_dict['q_9'])) + AVL_dict['q_10'])
    dmEGL19dt = (m_EGL19_inf - m_EGL19) / tau_m_EGL19
    dhEGL19dt = (h_EGL19_inf - h_EGL19) / tau_h_EGL19

    # differentials for I_CCA_1

    m_CCA1_inf = 1 / (1 + np.exp(-(V-AVL_dict['c_1'])/AVL_dict['c_2']))
    h_CCA1_inf = 1 / (1 + np.exp( (V-AVL_dict['d_1'])/AVL_dict['d_2']))
    tau_m_CCA1 = AVL_dict['c_3'] / (1 + np.exp(-(V-AVL_dict['c_4'])/AVL_dict['c_5'])) + AVL_dict['c_6']
    tau_h_CCA1 = AVL_dict['d_3'] / (1 + np.exp( (V-AVL_dict['d_4'])/AVL_dict['d_5'])) + AVL_dict['d_6']
    dmCCA1dt = (m_CCA1_inf - m_CCA1) / tau_m_CCA1
    dhCCA1dt = (h_CCA1_inf - h_CCA1) / tau_h_CCA1

    # differentials for I_SHL1

    tau_m_SHL1  = AVL_dict['a_m'] / (np.exp(-(V-AVL_dict['b_m'])/AVL_dict['c_m']) + np.exp((V-AVL_dict['d_m'])/AVL_dict['e_m'])) + AVL_dict['f_m']
    tau_hf_SHL1 = AVL_dict['a_hf'] / (1 + np.exp((V-AVL_dict['b_hf'])/AVL_dict['c_hf'])) + AVL_dict['d_hf']
    tau_hs_SHL1 = AVL_dict['a_hs'] / (1 + np.exp((V-AVL_dict['b_hs'])/AVL_dict['c_hs'])) + AVL_dict['d_hs']
    m_SHL1_inf  = 1 / (1 + np.exp(-(V-AVL_dict['v_1'])/AVL_dict['v_2']))
    h_SHL1_inf = 1 / (1 + np.exp( (V-AVL_dict['v_3'])/AVL_dict['v_4']))
    dmSHL1dt = (m_SHL1_inf - m_SHL1) / tau_m_SHL1
    dhfSHL1dt = (h_SHL1_inf - hf_SHL1) / tau_hf_SHL1
    dhsSHL1dt = (h_SHL1_inf - hs_SHL1) / tau_hs_SHL1

    # differentials for I_EGL36

    m_EGL36_inf = 1 / (1 + np.exp(-(V-AVL_dict['e_1'])/AVL_dict['e_2']))
    tau_mf_EGL36 = AVL_dict['t_f']
    tau_mm_EGL36 = AVL_dict['t_m']
    tau_ms_EGL36 = AVL_dict['t_s']
    dmfEGL36dt = (m_EGL36_inf - mf_EGL36) / tau_mf_EGL36
    dmmEGL36dt = (m_EGL36_inf - mm_EGL36) / tau_mm_EGL36
    dmsEGL36dt = (m_EGL36_inf - ms_EGL36) / tau_ms_EGL36

    # differentials for I_EXP_2

    alpha_1 = AVL_dict['p_1'] * np.exp(AVL_dict['p_2'] * V)
    beta_1 = AVL_dict['p_3'] * np.exp(-AVL_dict['p_4'] * V)
    K_f = AVL_dict['p_5']
    K_b = AVL_dict['p_6']  
    alpha_2 = AVL_dict['p_7'] * np.exp(AVL_dict['p_8']  * V)
    beta_2 = AVL_dict['p_9'] * np.exp(-AVL_dict['p_10'] * V)
    alpha_i = AVL_dict['p_11'] * np.exp(AVL_dict['p_12'] * V)
    beta_i = AVL_dict['p_13'] * np.exp(-AVL_dict['p_14'] * V)
    alpha_i2 = AVL_dict['p_15'] * np.exp(AVL_dict['p_16'] * V)
    psi = beta_2 * beta_i * alpha_i2 / (alpha_2 * alpha_i)
    dC1EXP2dt = beta_1 * C2_EXP2 - alpha_1 * C1_EXP2
    dC2EXP2dt = alpha_1 * C1_EXP2 + K_b * C3_EXP2 - (beta_1 + K_f) * C2_EXP2
    dC3EXP2dt = K_f * C2_EXP2 + psi * I_EXP2_ + beta_2 * O_EXP2 - (K_b + alpha_i2 + alpha_2) * C3_EXP2
    dOEXP2dt  = beta_i * I_EXP2_ + alpha_2 * C3_EXP2 - (beta_2 + alpha_i) * O_EXP2

    dV_AVL_vec = np.zeros(params_obj_neural['N'])
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

    np.put(dV_AVL_vec, AVL_dict['AVL_ind'], dV_AVL)
    np.put(dC_1_vec, AVL_dict['AVL_ind'], dC1EXP2dt)
    np.put(dC_2_vec, AVL_dict['AVL_ind'], dC2EXP2dt)
    np.put(dC_3_vec, AVL_dict['AVL_ind'], dC3EXP2dt)
    np.put(dO_vec, AVL_dict['AVL_ind'], dOEXP2dt)
    np.put(dm_UNC2_vec, AVL_dict['AVL_ind'], dmUNC2dt)
    np.put(dh_UNC2_vec, AVL_dict['AVL_ind'], dhUNC2dt)
    np.put(dm_EGL19_vec, AVL_dict['AVL_ind'], dmEGL19dt)
    np.put(dh_EGL19_vec, AVL_dict['AVL_ind'], dhEGL19dt)
    np.put(dm_CCA1_vec, AVL_dict['AVL_ind'], dmCCA1dt)
    np.put(dh_CCA1_vec, AVL_dict['AVL_ind'], dhCCA1dt)
    np.put(dm_SHL1_vec, AVL_dict['AVL_ind'], dmSHL1dt)
    np.put(dh_SHL1f_vec, AVL_dict['AVL_ind'], dhfSHL1dt)
    np.put(dh_SHL1s_vec, AVL_dict['AVL_ind'], dhsSHL1dt)
    np.put(dm_EGL36f_vec, AVL_dict['AVL_ind'], dmfEGL36dt)
    np.put(dm_EGL36m_vec, AVL_dict['AVL_ind'], dmmEGL36dt)
    np.put(dm_EGL36s_vec, AVL_dict['AVL_ind'], dmsEGL36dt)

    d_AVL_dyn_vec = np.concatenate([dV_AVL_vec, dC_1_vec, dC_2_vec, dC_3_vec, dO_vec, dm_UNC2_vec, dh_UNC2_vec, dm_EGL19_vec, dh_EGL19_vec, dm_CCA1_vec,
     dh_CCA1_vec, dm_SHL1_vec, dh_SHL1f_vec, dh_SHL1s_vec, dm_EGL36f_vec, dm_EGL36m_vec, dm_EGL36s_vec])

    return d_AVL_dyn_vec

def combine_baseline_nonlinear_currents(rhs_dict): # Needs to be simplified

    if params_obj_neural['nonlinear_AWA'] == True and params_obj_neural['nonlinear_AVL'] == False:

        AWA_dyn_vec = np.concatenate([rhs_dict['Vvec'], rhs_dict['y'][params_obj_neural['N'] * 2:]])
        dAWA_dyn_vec = AWA_nonlinear_current(AWA_dyn_vec)
        AWA_inds = params_obj_neural['AWA_nonlinear_params']['AWA_inds']

        dV = (-(rhs_dict['VsubEc'] + rhs_dict['GapCon'] + rhs_dict['SynapCon']) + rhs_dict['Input'])/params_obj_neural['C']

        dV[AWA_inds] = (-(dAWA_dyn_vec[:params_obj_neural['N']][AWA_inds] + rhs_dict['GapCon'][AWA_inds] + rhs_dict['SynapCon'][AWA_inds]) + rhs_dict['Input'][AWA_inds])/params_obj_neural['AWA_nonlinear_params']['C']
        dS = np.subtract(rhs_dict['SynRise'], rhs_dict['SynDrop'])

        return np.concatenate((dV, dS, dAWA_dyn_vec[params_obj_neural['N']:]))

    elif params_obj_neural['nonlinear_AWA'] == False and params_obj_neural['nonlinear_AVL'] == True:

        AVL_dyn_vec = np.concatenate([rhs_dict['Vvec'], rhs_dict['y'][params_obj_neural['N'] * 2:]])
        dAVL_dyn_vec = AVL_nonlinear_current(AVL_dyn_vec)
        AVL_ind = params_obj_neural['AVL_nonlinear_params']['AVL_ind']

        dV = (-(rhs_dict['VsubEc'] + rhs_dict['GapCon'] + rhs_dict['SynapCon']) + rhs_dict['Input'])/params_obj_neural['C']

        dV[AVL_ind] = (-(dAVL_dyn_vec[:params_obj_neural['N']][AVL_ind] + rhs_dict['GapCon'][AVL_ind] + rhs_dict['SynapCon'][AVL_ind]) + rhs_dict['Input'][AVL_ind])/params_obj_neural['AVL_nonlinear_params']['C']
        dS = np.subtract(rhs_dict['SynRise'], rhs_dict['SynDrop'])

        return np.concatenate((dV, dS, dAVL_dyn_vec[params_obj_neural['N']:]))

    elif params_obj_neural['nonlinear_AWA'] == True and params_obj_neural['nonlinear_AVL'] == True:

        AWA_dyn_vec = np.concatenate([rhs_dict['Vvec'], rhs_dict['y'][params_obj_neural['N'] * 2:9*params_obj_neural['N']]])
        AVL_dyn_vec = np.concatenate([rhs_dict['Vvec'], rhs_dict['y'][9*params_obj_neural['N']:]])

        dAWA_dyn_vec = AWA_nonlinear_current(AWA_dyn_vec)
        dAVL_dyn_vec = AVL_nonlinear_current(AVL_dyn_vec)
        AWA_inds = params_obj_neural['AWA_nonlinear_params']['AWA_inds']
        AVL_ind = params_obj_neural['AVL_nonlinear_params']['AVL_ind']

        dV = (-(rhs_dict['VsubEc'] + rhs_dict['GapCon'] + rhs_dict['SynapCon']) + rhs_dict['Input'])/params_obj_neural['C']

        dV[AWA_inds] = (-(dAWA_dyn_vec[:params_obj_neural['N']][AWA_inds] + rhs_dict['GapCon'][AWA_inds] + rhs_dict['SynapCon'][AWA_inds]) + rhs_dict['Input'][AWA_inds])/params_obj_neural['AWA_nonlinear_params']['C']
        dV[AVL_ind] = (-(dAVL_dyn_vec[:params_obj_neural['N']][AVL_ind] + rhs_dict['GapCon'][AVL_ind] + rhs_dict['SynapCon'][AVL_ind]) + rhs_dict['Input'][AVL_ind])/params_obj_neural['AVL_nonlinear_params']['C']
        dS = np.subtract(rhs_dict['SynRise'], rhs_dict['SynDrop'])

        return np.concatenate((dV, dS, dAWA_dyn_vec[params_obj_neural['N']:], dAVL_dyn_vec[params_obj_neural['N']:]))

    else:

        dV = (-(rhs_dict['VsubEc'] + rhs_dict['GapCon'] + rhs_dict['SynapCon']) + rhs_dict['Input'])/params_obj_neural['C']
        dS = np.subtract(rhs_dict['SynRise'], rhs_dict['SynDrop'])

        return np.concatenate((dV, dS))

#########################################################################################################################################################################
### STANDALONE FUNCTIONS FOR NEURON DYNAMICS ############################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

eps = 1e-12

def xinf(v_h, veq_h, k_h):

    return 1 / (1 + np.exp((veq_h - v_h) / k_h))

def xinf_torch(v_h, veq_h, k_h):

    x = (veq_h - v_h) / (k_h + eps)

    return torch.sigmoid(-x)

def run_network_CBM(param_vec, iext_amp, neuron_type):

    param_vec_iext = np.append(param_vec, iext_amp)
    param_vec_iext = np.array(param_vec_iext)

    initcond = param_vec_iext[[22, 18, 19, 20]]

    if neuron_type == 'cap':

        r = de.ODEProblem(cap_model_rhs_numba, initcond, (0, 50), param_vec_iext)

    else:

        r = de.ODEProblem(cat_model_rhs_numba, initcond, (0, 50), param_vec_iext)

    sol = de.solve(r, de.KenCarp47(autodiff=False), saveat = 0.1, save_everystep=False, abstol = 1e-8, reltol = 1e-8)

    t = sol.t
    traj = np.vstack(sol.u)

    return traj

def cap_model_rhs_numba(dy, y, p, t):

    v = y[0]
    mCa = y[1]
    mK = y[2]
    hK = y[3]

    Iext = p[23]

    dv = (-p[0] * mCa * (v - p[4]) - p[1] * xinf(v, p[8], p[12]) * (v - p[5]) - \
    p[2] * mK * hK * (v - p[5]) - p[3] * (v - p[6]) + Iext) / p[21]

    dmCa = (xinf(v, p[7], p[11]) - mCa)/p[15]
    dmK = (xinf(v, p[9], p[13]) - mK)/p[16]
    dhK = (xinf(v, p[10], p[14]) - hK)/p[17]

    dy[:] = [dv, dmCa, dmK, dhK]

def cat_model_rhs_numba(dy, y, p, t):

    v = y[0]
    mCa = y[1]
    hCa = y[2]
    mK = y[3]

    Iext = p[23]

    dv = (-p[0] * mCa * hCa * (v - p[4]) - p[1] * xinf(v, p[9], p[13]) * (v - p[5]) - \
    p[2] * mK * (v - p[5]) - p[3] * (v - p[6]) + Iext) / p[21]

    dmCa = (xinf(v, p[7], p[11]) - mCa)/p[15]
    dhCa = (xinf(v, p[8], p[12]) - hCa)/p[16]
    dmK = (xinf(v, p[10], p[14]) - mK)/p[17]

    dy[:] = [dv, dmCa, dhCa, dmK]

numba_cap_model_rhs = numba.jit(cap_model_rhs_numba)
numba_cat_model_rhs = numba.jit(cat_model_rhs_numba)

def CAP_V2I_torch_scaled(voltage_V, p_cap):

    v = voltage_V
    mCa_inf = xinf_torch(v, p_cap[:, 7], p_cap[:, 11])
    hKir_inf = xinf_torch(v, p_cap[:, 8], p_cap[:, 12])
    mK_inf = xinf_torch(v, p_cap[:, 9], p_cap[:, 13])
    hK_inf = xinf_torch(v, p_cap[:, 10], p_cap[:, 14])

    totalCurr = p_cap[:, 0] * mCa_inf * (v - p_cap[:, 4]) + p_cap[:, 1] * hKir_inf * (v - p_cap[:, 5]) + \
    p_cap[:, 2] * mK_inf * hK_inf * (v - p_cap[:, 5]) + p_cap[:, 3] * (v - p_cap[:, 6])

    return totalCurr

def CAT_V2I_torch_scaled(voltage_V, p_cat):

    v = voltage_V
    mCa_inf = xinf_torch(v, p_cat[:, 7], p_cat[:, 11])
    hCa_inf = xinf_torch(v, p_cat[:, 8], p_cat[:, 12])
    hKir_inf = xinf_torch(v, p_cat[:, 9], p_cat[:, 13])
    mK_inf = xinf_torch(v, p_cat[:, 10], p_cat[:, 14])

    totalCurr = p_cat[:, 0] * mCa_inf * hCa_inf * (v - p_cat[:, 4]) + p_cat[:, 1] * hKir_inf * (v - p_cat[:, 5]) + \
    p_cat[:, 2] * mK_inf * (v - p_cat[:, 5]) + p_cat[:, 3] * (v - p_cat[:, 6])

    return totalCurr

def cap_model_rhs_torch_scaled(V, y, p, iext):

    # V = (128, 500, 11)
    # y = (128, 500, 3, 11)
    # p = (128, 24, 1, 1)
    # iext = (128, 500, 11)

    mCa = y[:, :, 0, :]
    mK = y[:, :, 1, :]
    hK = y[:, :, 2, :]

    dV = (-p[:, 0] * mCa * (V - p[:, 4]) - p[:, 1] * xinf_torch(V, p[:, 8], p[:, 12]) * (V - p[:, 5]) - \
    p[:, 2] * mK * hK * (V - p[:, 5]) - p[:, 3] * (V - p[:, 6]) + iext) / (p[:, 21] + eps)

    dmCa = (xinf_torch(V, p[:, 7], p[:, 11]) - mCa)/(p[:, 15] + eps)
    dmK = (xinf_torch(V, p[:, 9], p[:, 13]) - mK)/(p[:, 16] + eps)
    dhK = (xinf_torch(V, p[:, 10], p[:, 14]) - hK)/(p[:, 17] + eps)

    return torch.cat([dV, dmCa, dmK, dhK], axis = 2)

def cat_model_rhs_torch_scaled(V, y, p, iext):

    # V = (128, 500, 11)
    # y = (128, 500, 3, 11)
    # p = (128, 24, 1, 1)
    # iext = (128, 500, 11)

    mCa = y[:, :, 0, :]
    hCa = y[:, :, 1, :]
    mK = y[:, :, 2, :]

    dV = (-p[:, 0] * mCa * hCa * (V - p[:, 4]) - p[:, 1] * xinf_torch(V, p[:, 9], p[:, 13]) * (V - p[:, 5]) - \
    p[:, 2] * mK * (V - p[:, 5]) - p[:, 3] * (V - p[:, 6]) + iext) / (p[:, 21] + eps)

    dmCa = (xinf_torch(V, p[:, 7], p[:, 11]) - mCa)/(p[:, 15] + eps)
    dhCa = (xinf_torch(V, p[:, 8], p[:, 12]) - hCa)/(p[:, 16] + eps)
    dmK = (xinf_torch(V, p[:, 10], p[:, 14]) - mK)/(p[:, 17] + eps)

    return torch.cat([dV, dmCa, dhCa, dmK], axis = 2)

#########################################################################################################################################################################
### STANDALONE FUNCTIONS FOR NEURON DYNAMICS GENERIC ####################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################
#########################################################################################################################################################################

@numba.extending.overload(np.heaviside)
def np_heaviside(x1, x2):
    def heaviside_impl(x1, x2):
        if x1 < 0:
            return 0.0
        elif x1 > 0:
            return 1.0
        else:
            return x2

    return heaviside_impl

def run_network_generic(param_vec, iext_amp):

    param_vec[-1] = iext_amp
    param_vec = np.array(param_vec)

    initcond = param_vec[185:-2]

    r = de.ODEProblem(numba_generic_model_rhs, initcond, (0, 7000), param_vec)

    sol = de.solve(r, de.KenCarp47(autodiff=False), saveat = 1., save_everystep=False, abstol = 1e-8, reltol = 1e-8)

    t = sol.t
    traj = np.vstack(sol.u)#[:, 0]

    return traj

def generic_model_rhs(dy, y, p, t):

    V = y[0]
    m_SHL1 = y[1]
    h_SHL1_f = y[2]
    h_SHL1_s = y[3]
    m_KVS1 = y[4]
    h_KVS1 = y[5]
    m_SHK1 = y[6]
    h_SHK1 = y[7]
    m_KQT3_f = y[8]
    m_KQT3_s = y[9]
    w_KQT3 = y[10]
    s_KQT3 = y[11]
    m_EGL2 = y[12]
    m_EGL36_f = y[13]
    m_EGL36_m = y[14]
    m_EGL36_s = y[15]
    m_IRK = y[16]
    m_EGL19 = y[17]
    h_EGL19 = y[18]
    m_UNC2 = y[19]
    h_UNC2 = y[20]
    m_CCA1 = y[21]
    h_CCA1 = y[22]
    m_SLO1_EGL19 = y[23]
    m_SLO1_UNC2 = y[24]
    m_SLO2_EGL19 = y[25]
    m_SLO2_UNC2 = y[26]
    m_KCNL = y[27]
    CA2_pos_m = y[28]

    # Voltage Gated Potassium Currents

    I_SHL1 = p[0] * m_SHL1**3 * (0.7 * h_SHL1_f + 0.3 * h_SHL1_s) * (V - p[181])                                                 # SHL1_g, VK
    I_KVS1 = p[31] * m_KVS1 * h_KVS1 * (V - p[181])                                                                              # KVS1_g, VK
    I_SHK1 = p[19] * m_SHK1 * h_SHK1 * (V - p[181])                                                                              # SHK1_g, VK
    I_KQT3 = p[44] * (0.7 * m_KQT3_f + 0.3 * m_KQT3_s) * w_KQT3 * s_KQT3 * (V - p[181])                                          # KQT3_g, Vk
    I_EGL2 = p[70] * m_EGL2 * (V - p[181])                                                                                       # EGL2_g, VK
    I_EGL36 = p[77] * (0.33 * m_EGL36_f + 0.36 * m_EGL36_m + 0.39 * m_EGL36_s) * (V - p[181])                                    # EGL36_g, VK
    I_IRK = p[83] * m_IRK * (V - p[181])                                                                                         # IRK_g, VK

    I_K = I_SHL1 + I_KVS1 + I_SHK1 + I_KQT3 + I_EGL2 + I_EGL36 + I_IRK

    # Voltage Gated Calcium Currents

    I_EGL19 = p[92] * m_EGL19 * h_EGL19 * (V - p[182])                                                                           # EGL19_g, VCa 
    I_UNC2 = p[118] * m_UNC2 * h_UNC2 * (V - p[182])                                                                             # UNC2_g, VCa
    I_CCA1 = p[134] * m_CCA1**2 * h_CCA1 * (V - p[182])                                                                          # CCA1_g, VCa

    I_C = I_EGL19 + I_UNC2 + I_CCA1

    # Calcium Regulated Potassium Currents

    I_SLO1_EGL19 = p[147] * m_SLO1_EGL19 * h_EGL19 * (V - p[181])                                                                # SLO1_g, VK
    I_SLO1_UNC2 = p[147] * m_SLO1_UNC2 * h_UNC2 * (V - p[181])                                                                   # SLO1_g, VK
    I_SLO2_EGL19 = p[156] * m_SLO2_EGL19 * h_EGL19 * (V - p[181])                                                                # SLO2_g, VK
    I_SLO2_UNC2 = p[156] * m_SLO2_UNC2 * h_UNC2 * (V - p[181])                                                                   # SLO2_g, VK
    I_KCNL = p[165] * m_KCNL * (V - p[181])                                                                                      # KCNL_g, VK

    I_KCa = I_SLO1_EGL19 + I_SLO1_UNC2 + I_SLO2_EGL19 + I_SLO2_UNC2 + I_KCNL

    # Leak Currents

    I_NCA = p[179] * (V - p[183])                                                                                                # NCA_g, VNa
    I_LEAK = p[180] * (V - p[184])                                                                                               # LEAK_g, VL

    I_L = I_NCA + I_LEAK

    # Calcium Concentration

    I_Ca = p[168] * np.abs((V - p[182])) * 1e-9 * 1e-3                                                                           # ICC_gsc, VCa
    CA2_pos_n = I_Ca / (8 * np.pi * p[169] * p[171] * p[170]) * np.exp(-p[169]/np.sqrt(p[171]/(p[172] * p[173]))) * 1e+6         # ICC_r, ICC_DCa, ICC_F, ICC_r, ICC_DCa, ICC_KB_pos, ICC_B_tot

    ICC_alpha = 1 / (2 * p[175] * p[170])                                                                                        # ICC_Vcell, ICC_F
    DCA2_pos_m_1 = (1 - np.heaviside(V - p[182], 0)) * (-(p[176] * ICC_alpha * I_C * 1e-6) - ((CA2_pos_m - p[178])/p[177]))      # VCa, ICC_f, ICC_Ca2_pos_m_eq, ICC_tau_Ca
    DCA2_pos_m_2 = np.heaviside(V - p[182], 0) * (-(CA2_pos_m - p[178])/p[177])                                                  # VCa, ICC_Ca2_pos_m_eq, ICC_tau_Ca

    # Calculate differentials

    SHL1_m_inf = 1 / (1 + np.exp(-(V - p[1])/p[2]))                                                                              # SHL1_m_inf_Veq, SHL1_m_inf_ka
    SHL1_tau_m = p[5] / (np.exp(-(V - p[6])/p[7]) + np.exp((V - p[8])/p[9])) + p[10]                                             # SHL1_tau_m_a, SHL1_tau_m_b, SHL1_tau_m_c, SHL1_tau_m_d, SHL1_tau_m_e, SHL1_tau_m_f
    SHL1_h_f_inf = 1 / (1 + np.exp((V - p[3])/p[4]))                                                                             # SHL1_h_inf_Veq, SHL1_h_inf_ki
    SHL1_h_s_inf = 1 / (1 + np.exp((V - p[3])/p[4]))                                                                             # SHL1_h_inf_Veq, SHL1_h_inf_ki
    SHL1_tau_h_f = p[11] / (1 + np.exp((V - p[12])/p[13])) + p[14]                                                               # SHL1_tau_h_f_a, SHL1_tau_h_f_b, SHL1_tau_h_f_c, SHL1_tau_h_f_d
    SHL1_tau_h_s = p[15] / (1 + np.exp((V - p[16])/p[17])) + p[18]                                                               # SHL1_tau_h_s_a, SHL1_tau_h_s_b, SHL1_tau_h_s_c, SHL1_tau_h_s_d

    KVS1_m_inf = 1 / (1 + np.exp(-(V - p[32])/p[33]))                                                                            # KVS1_m_inf_Veq, KVS1_m_inf_ka
    KVS1_h_inf = 1 / (1 + np.exp((V - p[34])/p[35]))                                                                             # KVS1_h_inf_Veq, KVS1_h_inf_ki
    KVS1_tau_m = p[36] / (1 + np.exp((V - p[37])/p[38])) + p[39]                                                                 # KVS1_tau_m_a, KVS1_tau_m_b, KVS1_tau_m_c, KVS1_tau_m_d
    KVS1_tau_h = p[40] / (1 + np.exp((V - p[41])/p[42])) + p[43]                                                                 # KVS1_tau_h_a, KVS1_tau_h_b, KVS1_tau_h_c, KVS1_tau_h_d

    SHK1_m_inf = 1 / (1 + np.exp(-(V - p[20])/p[21]))                                                                            # SHK1_m_inf_Veq, SHK1_m_inf_ka
    SHK1_h_inf = 1 / (1 + np.exp((V - p[22])/p[23]))                                                                             # SHK1_h_inf_Veq, SHK1_h_inf_ki
    SHK1_tau_m = p[24] / (np.exp(-(V - p[25])/p[26]) + np.exp((V - p[27]) / p[28])) + p[29]                                      # SHK1_tau_m_a, SHK1_tau_m_b, SHK1_tau_m_c, SHK1_tau_m_d, SHK1_tau_m_e, SHK1_tau_m_f
    SHK1_tau_h = p[30]                                                                                                           # SHK1_tau_h_a

    KQT3_m_f_inf = 1 / (1 + np.exp(-(V - p[45])/p[46]))                                                                          # KQT3_m_inf_Veq, KQT3_m_inf_ka
    KQT3_m_s_inf = 1 / (1 + np.exp(-(V - p[45])/p[46]))                                                                          # KQT3_m_inf_Veq, KQT3_m_inf_ka
    KQT3_tau_m_f = p[55] / (1 + ((V + p[56]) / p[57])**2)                                                                        # KQT3_tau_m_f_a, KQT3_tau_m_f_b, KQT3_tau_m_f_c
    KQT3_tau_m_s = p[58] + p[59] / (1 + 10**(-p[60] * (p[61] - V))) + p[62] / (1 + 10**(-p[63] * (p[64] + V)))                   # KQT3_tau_m_s_a, KQT3_tau_m_s_b, KQT3_tau_m_s_c, KQT3_tau_m_s_d, KQT3_tau_m_s_e, KQT3_tau_m_s_f, KQT3_tau_m_s_g
    KQT3_w_inf = p[49] + p[50] / (1 + np.exp((V - p[47])/p[48]))                                                                 # KQT3_w_inf_a, KQT3_w_inf_b, KQT3_w_inf_Veq, KQT3_w_inf_ki
    KQT3_s_inf = p[53] + p[54] / (1 + np.exp((V - p[51])/p[52]))                                                                 # KQT3_s_inf_a, KQT3_s_inf_b, KQT3_s_inf_Veq, KQT3_s_inf_ki
    KQT3_tau_w = p[65] + p[66] / (1 + ((V - p[67]) / p[68])**2)                                                                  # KQT3_tau_w_a, KQT3_tau_w_b, KQT3_tau_w_c, KQT3_tau_w_d
    KQT3_tau_s = p[69]                                                                                                           # KQT3_tau_s_a

    EGL2_m_inf = 1 / (1 + np.exp(-(V - p[71])/p[72]))                                                                            # EGL2_m_inf_Veq, EGL2_m_inf_ka
    EGL2_tau_m = p[73] / (1 + np.exp((V - p[74]) / p[75])) + p[76]                                                               # EGL2_tau_m_a, EGL2_tau_m_b, EGL2_tau_m_c, EGL2_tau_m_d

    EGL36_m_f_inf = 1 / (1 + np.exp(-(V - p[78])/p[79]))                                                                         # EGL36_m_inf_Veq, EGL36_m_inf_ka
    EGL36_m_m_inf = 1 / (1 + np.exp(-(V - p[78])/p[79]))                                                                         # EGL36_m_inf_Veq, EGL36_m_inf_ka
    EGL36_m_s_inf = 1 / (1 + np.exp(-(V - p[78])/p[79]))                                                                         # EGL36_m_inf_Veq, EGL36_m_inf_ka
    EGL36_tau_m_f = p[82]                                                                                                        # EGL36_tau_m_f_a
    EGL36_tau_m_m = p[81]                                                                                                        # EGL36_tau_m_m_a
    EGL36_tau_m_s = p[80]                                                                                                        # EGL36_tau_m_s_a

    IRK_m_inf = 1 / (1 + np.exp((V - p[84])/p[85]))                                                                              # IRK_m_inf_Veq, IRK_m_inf_ka
    IRK_tau_m = p[86] / (np.exp(-(V - p[87])/p[88]) + np.exp((V - p[89])/p[90])) + p[91]                                         # IRK_tau_m_a, IRK_tau_m_b, IRK_tau_m_c, IRK_tau_m_d, IRK_tau_m_e, IRK_tau_m_f

    EGL19_m_inf = 1 / (1 + np.exp(-(V - p[93])/p[94]))                                                                           # EGL19_m_inf_Veq, EGL19_m_inf_ka
    EGL19_tau_m = (p[103] * np.exp(-((V - p[104])/p[105])**2)) + (p[106] * np.exp(-((V - p[107])/p[108])**2)) + p[109]           # EGL19_tau_m_a, EGL19_tau_m_b, EGL19_tau_m_c, EGL19_tau_m_d, EGL19_tau_m_e, EGL19_tau_m_f, EGL19_tau_m_g
    EGL19_h_inf = (p[99] / (1 + np.exp(-(V - p[95])/p[96])) + p[100]) * (p[101] / (1 + np.exp((V - p[97])/p[98])) + p[102])      # EGL19_h_inf_a, EGL19_h_inf_Veq, EGL19_h_inf_ki, EGL19_h_inf_b, EGL19_h_inf_c, EGL19_h_inf_Veq_b, EGL19_h_inf_ki_b, EGL19_h_inf_d
    EGL19_tau_h = p[110] * (p[111] / (1 + np.exp((V-p[112])/p[113])) + p[114] / (1 + np.exp((V -p[115])/p[116])) + p[117])       # EGL19_tau_h_a, EGL19_tau_h_b, EGL19_tau_h_c, EGL19_tau_h_d, EGL19_tau_h_e, EGL19_tau_h_f, EGL19_tau_h_g, EGL19_tau_h_h

    UNC2_m_inf = 1 / (1 + np.exp(-(V - p[119])/p[120]))                                                                          # UNC2_m_inf_Veq, UNC2_m_inf_ka
    UNC2_tau_m = p[123] / (np.exp(-(V - p[124])/p[125]) + np.exp((V - p[124])/p[126])) + p[127]                                  # UNC2_tau_m_a, UNC2_tau_m_b, UNC2_tau_m_c, UNC2_tau_m_b, UNC2_tau_m_d, UNC2_tau_m_e
    UNC2_h_inf = 1 / (1 + np.exp((V - p[121])/p[122]))                                                                           # UNC2_h_inf_Veq, UNC2_h_inf_ki
    UNC2_tau_h = p[128] / (1 + np.exp(-(V-p[129])/p[130])) + p[131] / (1 + np.exp((V-p[132])/p[133]))                            # UNC2_tau_h_a, UNC2_tau_h_b, UNC2_tau_h_c, UNC2_tau_h_d, UNC2_tau_h_e, UNC2_tau_h_f

    CCA1_m_inf = 1 / (1 + np.exp(-(V - p[135])/p[136]))                                                                          # CCA1_m_inf_Veq, CCA1_m_inf_ka
    CCA1_h_inf = 1 / (1 + np.exp((V - p[137])/p[138]))                                                                           # CCA1_h_inf_Veq, CCA1_h_inf_ki
    CCA1_tau_m = p[139] / (1 + np.exp(-(V-p[140])/p[141])) + p[142]                                                              # CCA1_tau_m_a, CCA1_tau_m_b, CCA1_tau_m_c, CCA1_tau_m_d
    CCA1_tau_h = p[143] / (1 + np.exp((V-p[144])/p[145])) + p[146]                                                               # CCA1_tau_h_a, CCA1_tau_h_b, CCA1_tau_h_c, CCA1_tau_h_d

    EGL19_alpha = EGL19_m_inf/EGL19_tau_m
    EGL19_beta = 1/EGL19_tau_m - EGL19_alpha
    UNC2_alpha = UNC2_m_inf/UNC2_tau_m
    UNC2_beta = 1/UNC2_tau_m - UNC2_alpha

    SLO1_k_neg_o = (p[150] * np.exp(-p[148] * V)) * (1/(1 + (CA2_pos_n/p[154])**p[155]))                                         # SLO1_w0_neg, SLO1_wyx, SLO1_Kyx, SLO1_nyx
    SLO1_k_neg_c = (p[150] * np.exp(-p[148] * V)) * (1/(1 + (p[174]/p[154])**p[155]))                                            # SLO1_w0_neg, SLO1_wyx, ICC_Ca2_pos_n_ci, SLO1_Kyx, SLO1_nyx
    SLO1_k_pos_o = (p[151] * np.exp(-p[149] * V)) * (1/(1 + (p[152]/CA2_pos_n)**p[153]))                                         # SLO1_w0_pos, SLO1_wxy, SLO1_Kxy, SLO1_nxy

    SLO2_k_neg_o = (p[159] * np.exp(-p[157] * V)) * (1/(1 + (CA2_pos_n/p[163])**p[164]))                                         # SLO2_w0_neg, SLO2_wyx, SLO2_Kyx, SLO2_nyx
    SLO2_k_neg_c = (p[159] * np.exp(-p[157] * V)) * (1/(1 + (p[174]/p[163])**p[164]))                                            # SLO2_w0_neg, SLO2_wyx, ICC_Ca2_pos_n_ci, SLO2_Kyx, SLO2_nyx
    SLO2_k_pos_o = (p[160] * np.exp(-p[158] * V)) * (1/(1 + (p[161]/CA2_pos_n)**p[162]))                                         # SLO2_w0_pos, SLO2_wxy, SLO2_Kxy, SLO2_nxy

    SLO1_EGL19_m_inf = (m_EGL19 * SLO1_k_pos_o * (EGL19_alpha + EGL19_beta + SLO1_k_neg_c)) / ((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + EGL19_alpha) + EGL19_beta * SLO1_k_neg_c)
    SLO1_EGL19_tau_m = (EGL19_alpha + EGL19_beta + SLO1_k_neg_c) / ((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + EGL19_alpha) + EGL19_beta * SLO1_k_neg_c)
    SLO1_UNC2_m_inf = (m_UNC2 * SLO1_k_pos_o * (UNC2_alpha + UNC2_beta + SLO1_k_neg_c)) / ((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + UNC2_alpha) + UNC2_beta * SLO1_k_neg_c)
    SLO1_UNC2_tau_m = (UNC2_alpha + UNC2_beta + SLO1_k_neg_c) / ((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + UNC2_alpha) + UNC2_beta * SLO1_k_neg_c)

    SLO2_EGL19_m_inf = (m_EGL19 * SLO2_k_pos_o * (EGL19_alpha + EGL19_beta + SLO2_k_neg_c)) / ((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + EGL19_alpha) + EGL19_beta * SLO2_k_neg_c)
    SLO2_EGL19_tau_m = (EGL19_alpha + EGL19_beta + SLO2_k_neg_c) / ((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + EGL19_alpha) + EGL19_beta * SLO2_k_neg_c)
    SLO2_UNC2_m_inf = (m_UNC2 * SLO2_k_pos_o * (UNC2_alpha + UNC2_beta + SLO2_k_neg_c)) / ((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + UNC2_alpha) + UNC2_beta * SLO2_k_neg_c)
    SLO2_UNC2_tau_m = (UNC2_alpha + UNC2_beta + SLO2_k_neg_c) / ((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + UNC2_alpha) + UNC2_beta * SLO2_k_neg_c)

    KCNL_m_inf = CA2_pos_m / (p[166] + CA2_pos_m)                                                                                # KCNL_KCa
    KCNL_tau_m = p[167]                                                                                                          # KCNL_tau_m_a

    if t < 1000 or t > 6000:

        Iext = 0

    else:

        Iext = p[215]

    dV = (-(I_K + I_C + I_KCa + I_L) + Iext) / p[214]
    Dm_SHL1 = (SHL1_m_inf - m_SHL1) / SHL1_tau_m
    Dh_SHL1_f = (SHL1_h_f_inf - h_SHL1_f) / SHL1_tau_h_f
    Dh_SHL1_s = (SHL1_h_s_inf - h_SHL1_s) / SHL1_tau_h_s
    Dm_KVS1 = (KVS1_m_inf - m_KVS1) / KVS1_tau_m
    Dh_KVS1 = (KVS1_h_inf - h_KVS1) / KVS1_tau_h
    Dm_SHK1 = (SHK1_m_inf - m_SHK1) / SHK1_tau_m
    Dh_SHK1 = (SHK1_h_inf - h_SHK1) / SHK1_tau_h
    Dm_KQT3_f = (KQT3_m_f_inf - m_KQT3_f) / KQT3_tau_m_f
    Dm_KQT3_s = (KQT3_m_s_inf - m_KQT3_s) / KQT3_tau_m_s
    Dw_KQT3 = (KQT3_w_inf - w_KQT3) / KQT3_tau_w
    Ds_KQT3 = (KQT3_s_inf - s_KQT3) / KQT3_tau_s
    Dm_EGL2 = (EGL2_m_inf - m_EGL2) / EGL2_tau_m
    Dm_EGL36_f = (EGL36_m_f_inf - m_EGL36_f) / EGL36_tau_m_f
    Dm_EGL36_m = (EGL36_m_m_inf - m_EGL36_m) / EGL36_tau_m_m
    Dm_EGL36_s = (EGL36_m_s_inf - m_EGL36_s) / EGL36_tau_m_s
    Dm_IRK = (IRK_m_inf - m_IRK) / IRK_tau_m
    Dm_EGL19 = (EGL19_m_inf - m_EGL19) / EGL19_tau_m
    Dh_EGL19 = (EGL19_h_inf - h_EGL19) / EGL19_tau_h
    Dm_UNC2 = (UNC2_m_inf - m_UNC2) / UNC2_tau_m
    Dh_UNC2 = (UNC2_h_inf - h_UNC2) / UNC2_tau_h
    Dm_CCA1 = (CCA1_m_inf - m_CCA1) / CCA1_tau_m
    Dh_CCA1 = (CCA1_h_inf - h_CCA1) / CCA1_tau_h
    Dm_SLO1_EGL19 = (SLO1_EGL19_m_inf - m_SLO1_EGL19) / SLO1_EGL19_tau_m
    Dm_SLO1_UNC2 = (SLO1_UNC2_m_inf - m_SLO1_UNC2) / SLO1_UNC2_tau_m
    Dm_SLO2_EGL19 = (SLO2_EGL19_m_inf - m_SLO2_EGL19) / SLO2_EGL19_tau_m
    Dm_SLO2_UNC2 = (SLO2_UNC2_m_inf - m_SLO2_UNC2) / SLO2_UNC2_tau_m
    Dm_KCNL = (KCNL_m_inf - m_KCNL) / KCNL_tau_m
    DCA2_pos_m = DCA2_pos_m_1 + DCA2_pos_m_2

    dy[:] = np.array([dV, Dm_SHL1, Dh_SHL1_f, Dh_SHL1_s, Dm_KVS1, Dh_KVS1, Dm_SHK1, Dh_SHK1, Dm_KQT3_f, Dm_KQT3_s, Dw_KQT3, Ds_KQT3, Dm_EGL2, Dm_EGL36_f, Dm_EGL36_m, Dm_EGL36_s, 
        Dm_IRK, Dm_EGL19, Dh_EGL19, Dm_UNC2, Dh_UNC2, Dm_CCA1, Dh_CCA1, Dm_SLO1_EGL19, Dm_SLO1_UNC2, Dm_SLO2_EGL19, Dm_SLO2_UNC2, Dm_KCNL, DCA2_pos_m])

numba_generic_model_rhs = numba.jit(generic_model_rhs)

def generic_model_IV(V, p):

    # Calcium Concentration

    I_Ca = p[168] * np.abs((V - p[182])) * 1e-9 * 1e-3                                                                           # ICC_gsc, VCa
    CA2_pos_n = I_Ca / (8 * np.pi * p[169] * p[171] * p[170]) * np.exp(-p[169]/np.sqrt(p[171]/(p[172] * p[173]))) * 1e+6         # ICC_r, ICC_DCa, ICC_F, ICC_r, ICC_DCa, ICC_KB_pos, ICC_B_tot                

    # Calculate differentials

    SHL1_m_inf = 1 / (1 + np.exp(-(V - p[1])/p[2]))                                                                              # SHL1_m_inf_Veq, SHL1_m_inf_ka
    SHL1_h_f_inf = 1 / (1 + np.exp((V - p[3])/p[4]))                                                                             # SHL1_h_inf_Veq, SHL1_h_inf_ki
    SHL1_h_s_inf = 1 / (1 + np.exp((V - p[3])/p[4]))                                                                             # SHL1_h_inf_Veq, SHL1_h_inf_ki

    KVS1_m_inf = 1 / (1 + np.exp(-(V - p[32])/p[33]))                                                                            # KVS1_m_inf_Veq, KVS1_m_inf_ka
    KVS1_h_inf = 1 / (1 + np.exp((V - p[34])/p[35]))                                                                             # KVS1_h_inf_Veq, KVS1_h_inf_ki

    SHK1_m_inf = 1 / (1 + np.exp(-(V - p[20])/p[21]))                                                                            # SHK1_m_inf_Veq, SHK1_m_inf_ka
    SHK1_h_inf = 1 / (1 + np.exp((V - p[22])/p[23]))                                                                             # SHK1_h_inf_Veq, SHK1_h_inf_ki
    
    KQT3_m_f_inf = 1 / (1 + np.exp(-(V - p[45])/p[46]))                                                                          # KQT3_m_inf_Veq, KQT3_m_inf_ka
    KQT3_m_s_inf = 1 / (1 + np.exp(-(V - p[45])/p[46]))                                                                          # KQT3_m_inf_Veq, KQT3_m_inf_ka
    KQT3_w_inf = p[49] + p[50] / (1 + np.exp((V - p[47])/p[48]))                                                                 # KQT3_w_inf_a, KQT3_w_inf_b, KQT3_w_inf_Veq, KQT3_w_inf_ki
    KQT3_s_inf = p[53] + p[54] / (1 + np.exp((V - p[51])/p[52]))                                                                 # KQT3_s_inf_a, KQT3_s_inf_b, KQT3_s_inf_Veq, KQT3_s_inf_ki

    EGL2_m_inf = 1 / (1 + np.exp(-(V - p[71])/p[72]))                                                                            # EGL2_m_inf_Veq, EGL2_m_inf_ka

    EGL36_m_f_inf = 1 / (1 + np.exp(-(V - p[78])/p[79]))                                                                         # EGL36_m_inf_Veq, EGL36_m_inf_ka
    EGL36_m_m_inf = 1 / (1 + np.exp(-(V - p[78])/p[79]))                                                                         # EGL36_m_inf_Veq, EGL36_m_inf_ka
    EGL36_m_s_inf = 1 / (1 + np.exp(-(V - p[78])/p[79]))                                                                         # EGL36_m_inf_Veq, EGL36_m_inf_ka

    IRK_m_inf = 1 / (1 + np.exp((V - p[84])/p[85]))                                                                              # IRK_m_inf_Veq, IRK_m_inf_ka

    EGL19_m_inf = 1 / (1 + np.exp(-(V - p[93])/p[94]))                                                                           # EGL19_m_inf_Veq, EGL19_m_inf_ka
    EGL19_tau_m = (p[103] * np.exp(-((V - p[104])/p[105])**2)) + (p[106] * np.exp(-((V - p[107])/p[108])**2)) + p[109]           # EGL19_tau_m_a, EGL19_tau_m_b, EGL19_tau_m_c, EGL19_tau_m_d, EGL19_tau_m_e, EGL19_tau_m_f, EGL19_tau_m_g
    EGL19_h_inf = (p[99] / (1 + np.exp(-(V - p[95])/p[96])) + p[100]) * (p[101] / (1 + np.exp((V - p[97])/p[98])) + p[102])      # EGL19_h_inf_a, EGL19_h_inf_Veq, EGL19_h_inf_ki, EGL19_h_inf_b, EGL19_h_inf_c, EGL19_h_inf_Veq_b, EGL19_h_inf_ki_b, EGL19_h_inf_d

    UNC2_m_inf = 1 / (1 + np.exp(-(V - p[119])/p[120]))                                                                          # UNC2_m_inf_Veq, UNC2_m_inf_ka
    UNC2_tau_m = p[123] / (np.exp(-(V - p[124])/p[125]) + np.exp((V - p[124])/p[126])) + p[127]                                  # UNC2_tau_m_a, UNC2_tau_m_b, UNC2_tau_m_c, UNC2_tau_m_b, UNC2_tau_m_d, UNC2_tau_m_e
    UNC2_h_inf = 1 / (1 + np.exp((V - p[121])/p[122]))                                                                           # UNC2_h_inf_Veq, UNC2_h_inf_ki

    CCA1_m_inf = 1 / (1 + np.exp(-(V - p[135])/p[136]))                                                                          # CCA1_m_inf_Veq, CCA1_m_inf_ka
    CCA1_h_inf = 1 / (1 + np.exp((V - p[137])/p[138]))                                                                           # CCA1_h_inf_Veq, CCA1_h_inf_ki

    EGL19_alpha = EGL19_m_inf/EGL19_tau_m
    EGL19_beta = 1/EGL19_tau_m - EGL19_alpha
    UNC2_alpha = UNC2_m_inf/UNC2_tau_m
    UNC2_beta = 1/UNC2_tau_m - UNC2_alpha

    SLO1_k_neg_o = (p[150] * np.exp(-p[148] * V)) * (1/(1 + (CA2_pos_n/p[154])**p[155]))                                         # SLO1_w0_neg, SLO1_wyx, SLO1_Kyx, SLO1_nyx
    SLO1_k_neg_c = (p[150] * np.exp(-p[148] * V)) * (1/(1 + (p[174]/p[154])**p[155]))                                            # SLO1_w0_neg, SLO1_wyx, ICC_Ca2_pos_n_ci, SLO1_Kyx, SLO1_nyx
    SLO1_k_pos_o = (p[151] * np.exp(-p[149] * V)) * (1/(1 + (p[152]/CA2_pos_n)**p[153]))                                         # SLO1_w0_pos, SLO1_wxy, SLO1_Kxy, SLO1_nxy

    SLO2_k_neg_o = (p[159] * np.exp(-p[157] * V)) * (1/(1 + (CA2_pos_n/p[163])**p[164]))                                         # SLO2_w0_neg, SLO2_wyx, SLO2_Kyx, SLO2_nyx
    SLO2_k_neg_c = (p[159] * np.exp(-p[157] * V)) * (1/(1 + (p[174]/p[163])**p[164]))                                            # SLO2_w0_neg, SLO2_wyx, ICC_Ca2_pos_n_ci, SLO2_Kyx, SLO2_nyx
    SLO2_k_pos_o = (p[160] * np.exp(-p[158] * V)) * (1/(1 + (p[161]/CA2_pos_n)**p[162]))                                         # SLO2_w0_pos, SLO2_wxy, SLO2_Kxy, SLO2_nxy

    SLO1_EGL19_m_inf = (EGL19_m_inf * SLO1_k_pos_o * (EGL19_alpha + EGL19_beta + SLO1_k_neg_c)) / ((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + EGL19_alpha) + EGL19_beta * SLO1_k_neg_c)
    SLO1_UNC2_m_inf = (UNC2_m_inf * SLO1_k_pos_o * (UNC2_alpha + UNC2_beta + SLO1_k_neg_c)) / ((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + UNC2_alpha) + UNC2_beta * SLO1_k_neg_c)

    SLO2_EGL19_m_inf = (EGL19_m_inf * SLO2_k_pos_o * (EGL19_alpha + EGL19_beta + SLO2_k_neg_c)) / ((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + EGL19_alpha) + EGL19_beta * SLO2_k_neg_c)
    SLO2_UNC2_m_inf = (UNC2_m_inf * SLO2_k_pos_o * (UNC2_alpha + UNC2_beta + SLO2_k_neg_c)) / ((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + UNC2_alpha) + UNC2_beta * SLO2_k_neg_c)

    KCNL_m_inf = p[178] / (p[166] + p[178])                                                                                # KCNL_KCa                                                                                

    # Voltage Gated Potassium Currents

    I_SHL1 = p[0] * SHL1_m_inf**3 * (0.7 * SHL1_h_f_inf + 0.3 * SHL1_h_s_inf) * (V - p[181])                                                 
    I_KVS1 = p[31] * KVS1_m_inf * KVS1_h_inf * (V - p[181])                                                                              
    I_SHK1 = p[19] * SHK1_m_inf * SHK1_h_inf * (V - p[181])                                                                              
    I_KQT3 = p[44] * (0.7 * KQT3_m_f_inf + 0.3 * KQT3_m_s_inf) * KQT3_w_inf * KQT3_s_inf * (V - p[181])                                          
    I_EGL2 = p[70] * EGL2_m_inf * (V - p[181])                                                                                       
    I_EGL36 = p[77] * (0.33 * EGL36_m_f_inf + 0.36 * EGL36_m_m_inf + 0.39 * EGL36_m_s_inf) * (V - p[181])                                    
    I_IRK = p[83] * IRK_m_inf * (V - p[181])                                                                                         

    I_K = I_SHL1 + I_KVS1 + I_SHK1 + I_KQT3 + I_EGL2 + I_EGL36 + I_IRK

    # Voltage Gated Calcium Currents

    I_EGL19 = p[92] * EGL19_m_inf * EGL19_h_inf * (V - p[182])                                                                           
    I_UNC2 = p[118] * UNC2_m_inf * UNC2_h_inf * (V - p[182])                                                                             
    I_CCA1 = p[134] * CCA1_m_inf**2 * CCA1_h_inf * (V - p[182])                                                                          

    I_C = I_EGL19 + I_UNC2 + I_CCA1

    # Calcium Regulated Potassium Currents

    I_SLO1_EGL19 = p[147] * SLO1_EGL19_m_inf * EGL19_h_inf * (V - p[181])                                                                
    I_SLO1_UNC2 = p[147] * SLO1_UNC2_m_inf * UNC2_h_inf * (V - p[181])                                                                   
    I_SLO2_EGL19 = p[156] * SLO2_EGL19_m_inf * EGL19_h_inf * (V - p[181])                                                                
    I_SLO2_UNC2 = p[156] * SLO2_UNC2_m_inf * UNC2_h_inf * (V - p[181])                                                                   
    I_KCNL = p[165] * KCNL_m_inf * (V - p[181])                                                                                      

    I_KCa = I_SLO1_EGL19 + I_SLO1_UNC2 + I_SLO2_EGL19 + I_SLO2_UNC2 + I_KCNL

    # Leak Currents

    I_NCA = p[179] * (V - p[183])                                                                                                
    I_LEAK = p[180] * (V - p[184])                                                                                               

    I_L = I_NCA + I_LEAK

    totalCurr = I_K + I_C + I_KCa + I_L

    return totalCurr

def xinf_pos(x):

    return torch.sigmoid(-x)

def xinf_neg(x):

    return torch.sigmoid(x)

def generic_model_IV_torch_scaled(V, p):

    # Calcium Concentration

    I_Ca = p[:, 168] * torch.abs((V - p[:, 182])) * 1e-8 * 1e-1
    CA2_pos_n = I_Ca / (8 * torch.pi * p[:, 169] * p[:, 171] * p[:, 170]) * torch.exp(-p[:, 169]/torch.sqrt(p[:, 171]/(p[:, 172] * p[:, 173]))) * 1e+6

    # Calculate differentials
    SHL1_m_inf = xinf_neg((V - p[:, 1])/(p[:, 2] + eps))
    SHL1_h_f_inf = xinf_pos((V - p[:, 3])/(p[:, 4] + eps))
    SHL1_h_s_inf = xinf_pos((V - p[:, 3])/(p[:, 4] + eps))

    KVS1_m_inf = xinf_neg((V - p[:, 32])/(p[:, 33] + eps))
    KVS1_h_inf = xinf_pos((V - p[:, 34])/(p[:, 35] + eps))

    SHK1_m_inf = xinf_neg((V - p[:, 20])/(p[:, 21] + eps))
    SHK1_h_inf = xinf_pos((V - p[:, 22])/(p[:, 23] + eps))
    
    KQT3_m_f_inf = xinf_neg((V - p[:, 45])/(p[:, 46] + eps))
    KQT3_m_s_inf = xinf_neg((V - p[:, 45])/(p[:, 46] + eps))
    KQT3_w_inf = p[:, 49] + p[:, 50] * xinf_pos((V - p[:, 47])/(p[:, 48] + eps))
    KQT3_s_inf = p[:, 53] + p[:, 54] * xinf_pos((V - p[:, 51])/(p[:, 52] + eps))

    EGL2_m_inf = xinf_neg((V - p[:, 71])/(p[:, 72] + eps))

    EGL36_m_f_inf = xinf_neg((V - p[:, 78])/(p[:, 79] + eps))
    EGL36_m_m_inf = xinf_neg((V - p[:, 78])/(p[:, 79] + eps))
    EGL36_m_s_inf = xinf_neg((V - p[:, 78])/(p[:, 79] + eps))

    IRK_m_inf = xinf_pos((V - p[:, 84])/(p[:, 85] + eps))

    EGL19_m_inf = xinf_neg((V - p[:, 93])/(p[:, 94] + eps))
    EGL19_tau_m = (p[:, 103] * torch.exp(-((V - p[:, 104])/(p[:, 105] + eps))**2)) + (p[:, 106] * torch.exp(-((V - p[:, 107])/(p[:, 108] + eps))**2)) + p[:, 109]
    EGL19_h_inf = (p[:, 99] * xinf_neg((V - p[:, 95])/(p[:, 96] + eps)) + p[:, 100]) * (p[:, 101] * xinf_pos((V - p[:, 97])/(p[:, 98] + eps)) + p[:, 102])
    
    UNC2_m_inf = xinf_neg((V - p[:, 119])/(p[:, 120] + eps))
    UNC2_tau_m = p[:, 123] / (torch.exp(-(V - p[:, 124])/(p[:, 125] + eps)) + torch.exp((V - p[:, 124])/(p[:, 126] + eps))) + p[:, 127]
    UNC2_h_inf = xinf_pos((V - p[:, 121])/(p[:, 122] + eps))

    CCA1_m_inf = xinf_neg((V - p[:, 135])/(p[:, 136] + eps))
    CCA1_h_inf = xinf_pos((V - p[:, 137])/(p[:, 138] + eps))

    EGL19_alpha = EGL19_m_inf/(EGL19_tau_m + eps)
    EGL19_beta = 1/(EGL19_tau_m + eps) - EGL19_alpha
    UNC2_alpha = UNC2_m_inf/(UNC2_tau_m + eps)
    UNC2_beta = 1/(UNC2_tau_m + eps) - UNC2_alpha

    SLO1_k_neg_o = (p[:, 150] * torch.exp(-p[:, 148] * V)) * (1/(1 + (CA2_pos_n/p[:, 154])**p[:, 155]))
    SLO1_k_neg_c = (p[:, 150] * torch.exp(-p[:, 148] * V)) * (1/(1 + (p[:, 174]/p[:, 154])**p[:, 155]))           
    SLO1_k_pos_o = (p[:, 151] * torch.exp(-p[:, 149] * V)) * (1/(1 + (p[:, 152]/CA2_pos_n)**p[:, 153]))                   

    SLO2_k_neg_o = (p[:, 159] * torch.exp(-p[:, 157] * V)) * (1/(1 + (CA2_pos_n/p[:, 163])**p[:, 164]))
    SLO2_k_neg_c = (p[:, 159] * torch.exp(-p[:, 157] * V)) * (1/(1 + (p[:, 174]/p[:, 163])**p[:, 164]))                                        
    SLO2_k_pos_o = (p[:, 160] * torch.exp(-p[:, 158] * V)) * (1/(1 + (p[:, 161]/CA2_pos_n)**p[:, 162]))

    SLO1_EGL19_m_inf = (EGL19_m_inf * SLO1_k_pos_o * (EGL19_alpha + EGL19_beta + SLO1_k_neg_c)) / (((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + EGL19_alpha) + EGL19_beta * SLO1_k_neg_c) + eps)
    SLO1_UNC2_m_inf = (UNC2_m_inf * SLO1_k_pos_o * (UNC2_alpha + UNC2_beta + SLO1_k_neg_c)) / (((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + UNC2_alpha) + UNC2_beta * SLO1_k_neg_c) + eps)

    SLO2_EGL19_m_inf = (EGL19_m_inf * SLO2_k_pos_o * (EGL19_alpha + EGL19_beta + SLO2_k_neg_c)) / (((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + EGL19_alpha) + EGL19_beta * SLO2_k_neg_c) + eps)
    SLO2_UNC2_m_inf = (UNC2_m_inf * SLO2_k_pos_o * (UNC2_alpha + UNC2_beta + SLO2_k_neg_c)) / (((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + UNC2_alpha) + UNC2_beta * SLO2_k_neg_c) + eps)

    KCNL_m_inf = p[:, 178] / (p[:, 166] + p[:, 178] + eps)

    # Voltage Gated Potassium Currents

    I_SHL1 = p[:, 0] * SHL1_m_inf**3 * (0.7 * SHL1_h_f_inf + 0.3 * SHL1_h_s_inf) * (V - p[:, 181])                                                 
    I_KVS1 = p[:, 31] * KVS1_m_inf * KVS1_h_inf * (V - p[:, 181])                                                                              
    I_SHK1 = p[:, 19] * SHK1_m_inf * SHK1_h_inf * (V - p[:, 181])                                                                              
    I_KQT3 = p[:, 44] * (0.7 * KQT3_m_f_inf + 0.3 * KQT3_m_s_inf) * KQT3_w_inf * KQT3_s_inf * (V - p[:, 181])                                          
    I_EGL2 = p[:, 70] * EGL2_m_inf * (V - p[:, 181])                                                                                       
    I_EGL36 = p[:, 77] * (0.33 * EGL36_m_f_inf + 0.36 * EGL36_m_m_inf + 0.39 * EGL36_m_s_inf) * (V - p[:, 181])                                    
    I_IRK = p[:, 83] * IRK_m_inf * (V - p[:, 181])                                                                                         

    I_K = I_SHL1 + I_KVS1 + I_SHK1 + I_KQT3 + I_EGL2 + I_EGL36 + I_IRK

    # Voltage Gated Calcium Currents

    I_EGL19 = p[:, 92] * EGL19_m_inf * EGL19_h_inf * (V - p[:, 182])                                                                           
    I_UNC2 = p[:, 118] * UNC2_m_inf * UNC2_h_inf * (V - p[:, 182])                                                                             
    I_CCA1 = p[:, 134] * CCA1_m_inf**2 * CCA1_h_inf * (V - p[:, 182])                                                                          

    I_C = I_EGL19 + I_UNC2 + I_CCA1

    # Calcium Regulated Potassium Currents

    I_SLO1_EGL19 = p[:, 147] * SLO1_EGL19_m_inf * EGL19_h_inf * (V - p[:, 181])                                                                
    I_SLO1_UNC2 = p[:, 147] * SLO1_UNC2_m_inf * UNC2_h_inf * (V - p[:, 181])                                                                   
    I_SLO2_EGL19 = p[:, 156] * SLO2_EGL19_m_inf * EGL19_h_inf * (V - p[:, 181])                                                                
    I_SLO2_UNC2 = p[:, 156] * SLO2_UNC2_m_inf * UNC2_h_inf * (V - p[:, 181])                                                                   
    I_KCNL = p[:, 165] * KCNL_m_inf * (V - p[:, 181])                                                                                      

    I_KCa = I_SLO1_EGL19 + I_SLO1_UNC2 + I_SLO2_EGL19 + I_SLO2_UNC2 + I_KCNL

    # Leak Currents

    I_NCA = p[:, 179] * (V - p[:, 183])                                                                                                
    I_LEAK = p[:, 180] * (V - p[:, 184])                                                                                               

    I_L = I_NCA + I_LEAK

    totalCurr = I_K + I_C + I_KCa + I_L

    return totalCurr

def generic_model_IV_torch_unscaled(V, p):

    # Calcium Concentration

    I_Ca = p[:, 168] * torch.abs((V - p[:, 182])) * 1e-9 * 1e-3                                                                           # ICC_gsc, VCa
    CA2_pos_n = I_Ca / (8 * torch.pi * p[:, 169] * p[:, 171] * p[:, 170]) * torch.exp(-p[:, 169]/torch.sqrt(p[:, 171]/(p[:, 172] * p[:, 173]))) * 1e+6        # ICC_r, ICC_DCa, ICC_F, ICC_r, ICC_DCa, ICC_KB_pos, ICC_B_tot                

    # Calculate differentials

    SHL1_m_inf = 1 / (1 + torch.exp(-(V - p[:, 1])/p[:, 2]))                                                                              # SHL1_m_inf_Veq, SHL1_m_inf_ka
    SHL1_h_f_inf = 1 / (1 + torch.exp((V - p[:, 3])/p[:, 4]))                                                                             # SHL1_h_inf_Veq, SHL1_h_inf_ki
    SHL1_h_s_inf = 1 / (1 + torch.exp((V - p[:, 3])/p[:, 4]))                                                                             # SHL1_h_inf_Veq, SHL1_h_inf_ki

    KVS1_m_inf = 1 / (1 + torch.exp(-(V - p[:, 32])/p[:, 33]))                                                                            # KVS1_m_inf_Veq, KVS1_m_inf_ka
    KVS1_h_inf = 1 / (1 + torch.exp((V - p[:, 34])/p[:, 35]))                                                                             # KVS1_h_inf_Veq, KVS1_h_inf_ki

    SHK1_m_inf = 1 / (1 + torch.exp(-(V - p[:, 20])/p[:, 21]))                                                                            # SHK1_m_inf_Veq, SHK1_m_inf_ka
    SHK1_h_inf = 1 / (1 + torch.exp((V - p[:, 22])/p[:, 23]))                                                                             # SHK1_h_inf_Veq, SHK1_h_inf_ki
    
    KQT3_m_f_inf = 1 / (1 + torch.exp(-(V - p[:, 45])/p[:, 46]))                                                                          # KQT3_m_inf_Veq, KQT3_m_inf_ka
    KQT3_m_s_inf = 1 / (1 + torch.exp(-(V - p[:, 45])/p[:, 46]))                                                                          # KQT3_m_inf_Veq, KQT3_m_inf_ka
    KQT3_w_inf = p[:, 49] + p[:, 50] / (1 + torch.exp((V - p[:, 47])/p[:, 48]))                                                                 # KQT3_w_inf_a, KQT3_w_inf_b, KQT3_w_inf_Veq, KQT3_w_inf_ki
    KQT3_s_inf = p[:, 53] + p[:, 54] / (1 + torch.exp((V - p[:, 51])/p[:, 52]))                                                                 # KQT3_s_inf_a, KQT3_s_inf_b, KQT3_s_inf_Veq, KQT3_s_inf_ki

    EGL2_m_inf = 1 / (1 + torch.exp(-(V - p[:, 71])/p[:, 72]))                                                                            # EGL2_m_inf_Veq, EGL2_m_inf_ka

    EGL36_m_f_inf = 1 / (1 + torch.exp(-(V - p[:, 78])/p[:, 79]))                                                                         # EGL36_m_inf_Veq, EGL36_m_inf_ka
    EGL36_m_m_inf = 1 / (1 + torch.exp(-(V - p[:, 78])/p[:, 79]))                                                                         # EGL36_m_inf_Veq, EGL36_m_inf_ka
    EGL36_m_s_inf = 1 / (1 + torch.exp(-(V - p[:, 78])/p[:, 79]))                                                                         # EGL36_m_inf_Veq, EGL36_m_inf_ka

    IRK_m_inf = 1 / (1 + torch.exp((V - p[:, 84])/p[:, 85]))                                                                              # IRK_m_inf_Veq, IRK_m_inf_ka

    EGL19_m_inf = 1 / (1 + torch.exp(-(V - p[:, 93])/p[:, 94]))                                                                           # EGL19_m_inf_Veq, EGL19_m_inf_ka
    EGL19_tau_m = (p[:, 103] * torch.exp(-((V - p[:, 104])/p[:, 105])**2)) + (p[:, 106] * torch.exp(-((V - p[:, 107])/p[:, 108])**2)) + p[:, 109]           # EGL19_tau_m_a, EGL19_tau_m_b, EGL19_tau_m_c, EGL19_tau_m_d, EGL19_tau_m_e, EGL19_tau_m_f, EGL19_tau_m_g
    EGL19_h_inf = (p[:, 99] / (1 + torch.exp(-(V - p[:, 95])/p[:, 96])) + p[:, 100]) * (p[:, 101] / (1 + torch.exp((V - p[:, 97])/p[:, 98])) + p[:, 102])      # EGL19_h_inf_a, EGL19_h_inf_Veq, EGL19_h_inf_ki, EGL19_h_inf_b, EGL19_h_inf_c, EGL19_h_inf_Veq_b, EGL19_h_inf_ki_b, EGL19_h_inf_d

    UNC2_m_inf = 1 / (1 + torch.exp(-(V - p[:, 119])/p[:, 120]))                                                                          # UNC2_m_inf_Veq, UNC2_m_inf_ka
    UNC2_tau_m = p[:, 123] / (torch.exp(-(V - p[:, 124])/p[:, 125]) + torch.exp((V - p[:, 124])/p[:, 126])) + p[:, 127]                                  # UNC2_tau_m_a, UNC2_tau_m_b, UNC2_tau_m_c, UNC2_tau_m_b, UNC2_tau_m_d, UNC2_tau_m_e
    UNC2_h_inf = 1 / (1 + torch.exp((V - p[:, 121])/p[:, 122]))                                                                           # UNC2_h_inf_Veq, UNC2_h_inf_ki

    CCA1_m_inf = 1 / (1 + torch.exp(-(V - p[:, 135])/p[:, 136]))                                                                          # CCA1_m_inf_Veq, CCA1_m_inf_ka
    CCA1_h_inf = 1 / (1 + torch.exp((V - p[:, 137])/p[:, 138]))                                                                           # CCA1_h_inf_Veq, CCA1_h_inf_ki

    EGL19_alpha = EGL19_m_inf/(EGL19_tau_m + eps)
    EGL19_beta = 1/(EGL19_tau_m + eps) - EGL19_alpha
    UNC2_alpha = UNC2_m_inf/(UNC2_tau_m + eps)
    UNC2_beta = 1/(UNC2_tau_m + eps) - UNC2_alpha

    SLO1_k_neg_o = (p[:, 150] * torch.exp(-p[:, 148] * V)) * (1/(1 + (CA2_pos_n/p[:, 154])**p[:, 155]))                                         # SLO1_w0_neg, SLO1_wyx, SLO1_Kyx, SLO1_nyx
    SLO1_k_neg_c = (p[:, 150] * torch.exp(-p[:, 148] * V)) * (1/(1 + (p[:, 174]/p[:, 154])**p[:, 155]))                                            # SLO1_w0_neg, SLO1_wyx, ICC_Ca2_pos_n_ci, SLO1_Kyx, SLO1_nyx
    SLO1_k_pos_o = (p[:, 151] * torch.exp(-p[:, 149] * V)) * (1/(1 + (p[:, 152]/CA2_pos_n)**p[:, 153]))                                         # SLO1_w0_pos, SLO1_wxy, SLO1_Kxy, SLO1_nxy

    SLO2_k_neg_o = (p[:, 159] * torch.exp(-p[:, 157] * V)) * (1/(1 + (CA2_pos_n/p[:, 163])**p[:, 164]))                                         # SLO2_w0_neg, SLO2_wyx, SLO2_Kyx, SLO2_nyx
    SLO2_k_neg_c = (p[:, 159] * torch.exp(-p[:, 157] * V)) * (1/(1 + (p[:, 174]/p[:, 163])**p[:, 164]))                                            # SLO2_w0_neg, SLO2_wyx, ICC_Ca2_pos_n_ci, SLO2_Kyx, SLO2_nyx
    SLO2_k_pos_o = (p[:, 160] * torch.exp(-p[:, 158] * V)) * (1/(1 + (p[:, 161]/CA2_pos_n)**p[:, 162]))                                         # SLO2_w0_pos, SLO2_wxy, SLO2_Kxy, SLO2_nxy

    SLO1_EGL19_m_inf = (EGL19_m_inf * SLO1_k_pos_o * (EGL19_alpha + EGL19_beta + SLO1_k_neg_c)) / (((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + EGL19_alpha) + EGL19_beta * SLO1_k_neg_c) + eps)
    SLO1_UNC2_m_inf = (UNC2_m_inf * SLO1_k_pos_o * (UNC2_alpha + UNC2_beta + SLO1_k_neg_c)) / (((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + UNC2_alpha) + UNC2_beta * SLO1_k_neg_c) + eps)

    SLO2_EGL19_m_inf = (EGL19_m_inf * SLO2_k_pos_o * (EGL19_alpha + EGL19_beta + SLO2_k_neg_c)) / (((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + EGL19_alpha) + EGL19_beta * SLO2_k_neg_c) + eps)
    SLO2_UNC2_m_inf = (UNC2_m_inf * SLO2_k_pos_o * (UNC2_alpha + UNC2_beta + SLO2_k_neg_c)) / (((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + UNC2_alpha) + UNC2_beta * SLO2_k_neg_c) + eps)

    KCNL_m_inf = p[:, 178] / (p[:, 166] + p[:, 178])                                                                                # KCNL_KCa                                                                                

    # Voltage Gated Potassium Currents

    I_SHL1 = p[:, 0] * SHL1_m_inf**3 * (0.7 * SHL1_h_f_inf + 0.3 * SHL1_h_s_inf) * (V - p[:, 181])                                                 
    I_KVS1 = p[:, 31] * KVS1_m_inf * KVS1_h_inf * (V - p[:, 181])                                                                              
    I_SHK1 = p[:, 19] * SHK1_m_inf * SHK1_h_inf * (V - p[:, 181])                                                                              
    I_KQT3 = p[:, 44] * (0.7 * KQT3_m_f_inf + 0.3 * KQT3_m_s_inf) * KQT3_w_inf * KQT3_s_inf * (V - p[:, 181])                                          
    I_EGL2 = p[:, 70] * EGL2_m_inf * (V - p[:, 181])                                                                                       
    I_EGL36 = p[:, 77] * (0.33 * EGL36_m_f_inf + 0.36 * EGL36_m_m_inf + 0.39 * EGL36_m_s_inf) * (V - p[:, 181])                                    
    I_IRK = p[:, 83] * IRK_m_inf * (V - p[:, 181])                                                                                         

    I_K = I_SHL1 + I_KVS1 + I_SHK1 + I_KQT3 + I_EGL2 + I_EGL36 + I_IRK

    # Voltage Gated Calcium Currents

    I_EGL19 = p[:, 92] * EGL19_m_inf * EGL19_h_inf * (V - p[:, 182])                                                                           
    I_UNC2 = p[:, 118] * UNC2_m_inf * UNC2_h_inf * (V - p[:, 182])                                                                             
    I_CCA1 = p[:, 134] * CCA1_m_inf**2 * CCA1_h_inf * (V - p[:, 182])                                                                          

    I_C = I_EGL19 + I_UNC2 + I_CCA1

    # Calcium Regulated Potassium Currents

    I_SLO1_EGL19 = p[:, 147] * SLO1_EGL19_m_inf * EGL19_h_inf * (V - p[:, 181])                                                                
    I_SLO1_UNC2 = p[:, 147] * SLO1_UNC2_m_inf * UNC2_h_inf * (V - p[:, 181])                                                                   
    I_SLO2_EGL19 = p[:, 156] * SLO2_EGL19_m_inf * EGL19_h_inf * (V - p[:, 181])                                                                
    I_SLO2_UNC2 = p[:, 156] * SLO2_UNC2_m_inf * UNC2_h_inf * (V - p[:, 181])                                                                   
    I_KCNL = p[:, 165] * KCNL_m_inf * (V - p[:, 181])                                                                                      

    I_KCa = I_SLO1_EGL19 + I_SLO1_UNC2 + I_SLO2_EGL19 + I_SLO2_UNC2 + I_KCNL

    # Leak Currents

    I_NCA = p[:, 179] * (V - p[:, 183])                                                                                                
    I_LEAK = p[:, 180] * (V - p[:, 184])                                                                                               

    I_L = I_NCA + I_LEAK

    totalCurr = I_K + I_C + I_KCa + I_L

    return totalCurr

def generic_model_rhs_torch_scaled_vectorized(V, y, p, iext):

    # V = (128, 500, 11)
    # y = (128, 500, 28, 11)
    # p = (128, 216, 1, 1)
    # iext = (128, 500, 11)

    m_SHL1 = y[:, :, 0, :]
    h_SHL1_f = y[:, :, 1, :]
    h_SHL1_s = y[:, :, 2, :]
    m_KVS1 = y[:, :, 3, :]
    h_KVS1 = y[:, :, 4, :]
    m_SHK1 = y[:, :, 5, :]
    h_SHK1 = y[:, :, 6, :]
    m_KQT3_f = y[:, :, 7, :]
    m_KQT3_s = y[:, :, 8, :]
    w_KQT3 = y[:, :, 9, :]
    s_KQT3 = y[:, :, 10, :]
    m_EGL2 = y[:, :, 11, :]
    m_EGL36_f = y[:, :, 12, :]
    m_EGL36_m = y[:, :, 13, :]
    m_EGL36_s = y[:, :, 14, :]
    m_IRK = y[:, :, 15, :]
    m_EGL19 = y[:, :, 16, :]
    h_EGL19 = y[:, :, 17, :]
    m_UNC2 = y[:, :, 18, :]
    h_UNC2 = y[:, :, 19, :]
    m_CCA1 = y[:, :, 20, :]
    h_CCA1 = y[:, :, 21, :]
    m_SLO1_EGL19 = y[:, :, 22, :]
    m_SLO1_UNC2 = y[:, :, 23, :]
    m_SLO2_EGL19 = y[:, :, 24, :]
    m_SLO2_UNC2 = y[:, :, 25, :]
    m_KCNL = y[:, :, 26, :]
    CA2_pos_m = y[:, :, 27, :]

    # Voltage Gated Potassium Currents

    I_SHL1 = p[:, 0] * m_SHL1**3 * (0.7 * h_SHL1_f + 0.3 * h_SHL1_s) * (V - p[:, 181])                                                 # SHL1_g, VK
    I_KVS1 = p[:, 31] * m_KVS1 * h_KVS1 * (V - p[:, 181])                                                                              # KVS1_g, VK
    I_SHK1 = p[:, 19] * m_SHK1 * h_SHK1 * (V - p[:, 181])                                                                              # SHK1_g, VK
    I_KQT3 = p[:, 44] * (0.7 * m_KQT3_f + 0.3 * m_KQT3_s) * w_KQT3 * s_KQT3 * (V - p[:, 181])                                          # KQT3_g, Vk
    I_EGL2 = p[:, 70] * m_EGL2 * (V - p[:, 181])                                                                                       # EGL2_g, VK
    I_EGL36 = p[:, 77] * (0.33 * m_EGL36_f + 0.36 * m_EGL36_m + 0.39 * m_EGL36_s) * (V - p[:, 181])                                    # EGL36_g, VK
    I_IRK = p[:, 83] * m_IRK * (V - p[:, 181])                                                                                         # IRK_g, VK

    I_K = I_SHL1 + I_KVS1 + I_SHK1 + I_KQT3 + I_EGL2 + I_EGL36 + I_IRK

    # Voltage Gated Calcium Currents

    I_EGL19 = p[:, 92] * m_EGL19 * h_EGL19 * (V - p[:, 182])                                                                           # EGL19_g, VCa 
    I_UNC2 = p[:, 118] * m_UNC2 * h_UNC2 * (V - p[:, 182])                                                                             # UNC2_g, VCa
    I_CCA1 = p[:, 134] * m_CCA1**2 * h_CCA1 * (V - p[:, 182])                                                                          # CCA1_g, VCa

    I_C = I_EGL19 + I_UNC2 + I_CCA1

    # Calcium Regulated Potassium Currents

    I_SLO1_EGL19 = p[:, 147] * m_SLO1_EGL19 * h_EGL19 * (V - p[:, 181])                                                                # SLO1_g, VK
    I_SLO1_UNC2 = p[:, 147] * m_SLO1_UNC2 * h_UNC2 * (V - p[:, 181])                                                                   # SLO1_g, VK
    I_SLO2_EGL19 = p[:, 156] * m_SLO2_EGL19 * h_EGL19 * (V - p[:, 181])                                                                # SLO2_g, VK
    I_SLO2_UNC2 = p[:, 156] * m_SLO2_UNC2 * h_UNC2 * (V - p[:, 181])                                                                   # SLO2_g, VK
    I_KCNL = p[:, 165] * m_KCNL * (V - p[:, 181])                                                                                      # KCNL_g, VK

    I_KCa = I_SLO1_EGL19 + I_SLO1_UNC2 + I_SLO2_EGL19 + I_SLO2_UNC2 + I_KCNL

    # Leak Currents

    I_NCA = p[:, 179] * (V - p[:, 183])                                                                                                # NCA_g, VNa
    I_LEAK = p[:, 180] * (V - p[:, 184])                                                                                               # LEAK_g, VL

    I_L = I_NCA + I_LEAK                                                                                                               # KCNL_tau_m_a

    dV = (-(I_K + I_C + I_KCa + I_L) + iext) / p[:, 214]

    return dV

def generic_model_rhs_torch_scaled_vectorized_full(V, y, p, iext):

    # V = (128, 500, 11)
    # y = (128, 500, 28, 11)
    # p = (128, 216, 1, 1)
    # iext = (128, 500, 11)

    m_SHL1 = y[:, :, 0, :]
    h_SHL1_f = y[:, :, 1, :]
    h_SHL1_s = y[:, :, 2, :]
    m_KVS1 = y[:, :, 3, :]
    h_KVS1 = y[:, :, 4, :]
    m_SHK1 = y[:, :, 5, :]
    h_SHK1 = y[:, :, 6, :]
    m_KQT3_f = y[:, :, 7, :]
    m_KQT3_s = y[:, :, 8, :]
    w_KQT3 = y[:, :, 9, :]
    s_KQT3 = y[:, :, 10, :]
    m_EGL2 = y[:, :, 11, :]
    m_EGL36_f = y[:, :, 12, :]
    m_EGL36_m = y[:, :, 13, :]
    m_EGL36_s = y[:, :, 14, :]
    m_IRK = y[:, :, 15, :]
    m_EGL19 = y[:, :, 16, :]
    h_EGL19 = y[:, :, 17, :]
    m_UNC2 = y[:, :, 18, :]
    h_UNC2 = y[:, :, 19, :]
    m_CCA1 = y[:, :, 20, :]
    h_CCA1 = y[:, :, 21, :]
    m_SLO1_EGL19 = y[:, :, 22, :]
    m_SLO1_UNC2 = y[:, :, 23, :]
    m_SLO2_EGL19 = y[:, :, 24, :]
    m_SLO2_UNC2 = y[:, :, 25, :]
    m_KCNL = y[:, :, 26, :]
    CA2_pos_m = y[:, :, 27, :]

    # Voltage Gated Potassium Currents

    I_SHL1 = p[:, 0] * m_SHL1**3 * (0.7 * h_SHL1_f + 0.3 * h_SHL1_s) * (V - p[:, 181])                                                 
    I_KVS1 = p[:, 31] * m_KVS1 * h_KVS1 * (V - p[:, 181])                                                                              
    I_SHK1 = p[:, 19] * m_SHK1 * h_SHK1 * (V - p[:, 181])                                                                              
    I_KQT3 = p[:, 44] * (0.7 * m_KQT3_f + 0.3 * m_KQT3_s) * w_KQT3 * s_KQT3 * (V - p[:, 181])                                          
    I_EGL2 = p[:, 70] * m_EGL2 * (V - p[:, 181])                                                                                       
    I_EGL36 = p[:, 77] * (0.33 * m_EGL36_f + 0.36 * m_EGL36_m + 0.39 * m_EGL36_s) * (V - p[:, 181])                                    
    I_IRK = p[:, 83] * m_IRK * (V - p[:, 181])                                                                                         

    I_K = I_SHL1 + I_KVS1 + I_SHK1 + I_KQT3 + I_EGL2 + I_EGL36 + I_IRK

    # Voltage Gated Calcium Currents

    I_EGL19 = p[:, 92] * m_EGL19 * h_EGL19 * (V - p[:, 182])                                                                           
    I_UNC2 = p[:, 118] * m_UNC2 * h_UNC2 * (V - p[:, 182])                                                                             
    I_CCA1 = p[:, 134] * m_CCA1**2 * h_CCA1 * (V - p[:, 182])                                                                          

    I_C = I_EGL19 + I_UNC2 + I_CCA1

    # Calcium Regulated Potassium Currents

    I_SLO1_EGL19 = p[:, 147] * m_SLO1_EGL19 * h_EGL19 * (V - p[:, 181])                                                                
    I_SLO1_UNC2 = p[:, 147] * m_SLO1_UNC2 * h_UNC2 * (V - p[:, 181])                                                                   
    I_SLO2_EGL19 = p[:, 156] * m_SLO2_EGL19 * h_EGL19 * (V - p[:, 181])                                                                
    I_SLO2_UNC2 = p[:, 156] * m_SLO2_UNC2 * h_UNC2 * (V - p[:, 181])                                                                   
    I_KCNL = p[:, 165] * m_KCNL * (V - p[:, 181])                                                                                      

    I_KCa = I_SLO1_EGL19 + I_SLO1_UNC2 + I_SLO2_EGL19 + I_SLO2_UNC2 + I_KCNL

    # Leak Currents

    I_NCA = p[:, 179] * (V - p[:, 183])                                                                                                
    I_LEAK = p[:, 180] * (V - p[:, 184])                                                                                               

    I_L = I_NCA + I_LEAK                                                                                                               

    # Calcium Concentration

    I_Ca = p[:, 168] * torch.abs((V - p[:, 182])) * 1e-8 * 1e-1                                                                           
    CA2_pos_n = I_Ca / (8 * torch.pi * p[:, 169] * p[:, 171] * p[:, 170]) * torch.exp(-p[:, 169]/torch.sqrt(p[:, 171]/(p[:, 172] * p[:, 173]))) * 1e+6         

    ICC_alpha = 1 / (2 * p[:, 175] * p[:, 170])                                                                                        
    #DCA2_pos_m_1 = (1 - torch.heaviside(V - p[:, 182], torch.tensor([0.]))) * (-(p[:, 176] * ICC_alpha * I_C * 1e-3) - ((CA2_pos_m - p[:, 178])/p[:, 177]))      
    #DCA2_pos_m_2 = torch.heaviside(V - p[:, 182], torch.tensor([0.])) * (-(CA2_pos_m - p[:, 178])/p[:, 177])
    DCA2_pos_m_1 = torch.zeros((len(y), 500, 11)).float().cuda() 
    DCA2_pos_m_2 = torch.zeros((len(y), 500, 11)).float().cuda()                                                   

    # Calculate differentials

    SHL1_m_inf = xinf_neg((V - p[:, 1])/(p[:, 2] + eps))                                                                              
    SHL1_tau_m = p[:, 5] / (torch.exp(-(V - p[:, 6])/(p[:, 7] + eps)) + torch.exp((V - p[:, 8])/(p[:, 9] + eps))) + p[:, 10] # 6,7,8,9                                            
    SHL1_h_f_inf = xinf_pos((V - p[:, 3])/(p[:, 4] + eps))                                                                             
    SHL1_h_s_inf = xinf_pos((V - p[:, 3])/(p[:, 4] + eps))                                                                             
    SHL1_tau_h_f = p[:, 11] * xinf_pos((V - p[:, 12])/(p[:, 13] + eps)) + p[:, 14]                                                  
    SHL1_tau_h_s = p[:, 15] * xinf_pos((V - p[:, 16])/(p[:, 17] + eps)) + p[:, 18]                                                               

    KVS1_m_inf = xinf_neg((V - p[:, 32])/(p[:, 33] + eps))                                                                            
    KVS1_h_inf = xinf_pos((V - p[:, 34])/(p[:, 35] + eps))                                                                             
    KVS1_tau_m = p[:, 36] * xinf_pos((V - p[:, 37])/(p[:, 38] + eps)) + p[:, 39]                                                               
    KVS1_tau_h = p[:, 40] * xinf_pos((V - p[:, 41])/(p[:, 42] + eps)) + p[:, 43]                                                               

    SHK1_m_inf = xinf_neg((V - p[:, 20])/(p[:, 21] + eps))                                                                            
    SHK1_h_inf = xinf_pos((V - p[:, 22])/(p[:, 23] + eps))                                                                             
    SHK1_tau_m = p[:, 24] / (torch.exp(-(V - p[:, 25])/(p[:, 26] + eps)) + torch.exp((V - p[:, 27])/(p[:, 28] + eps))) + p[:, 29] # 25,26,27,28                                      
    SHK1_tau_h = p[:, 30]                                                                                                           

    KQT3_m_f_inf = xinf_neg((V - p[:, 45])/(p[:, 46] + eps))                                                                          
    KQT3_m_s_inf = xinf_neg((V - p[:, 45])/(p[:, 46] + eps))                                                                          
    KQT3_tau_m_f = p[:, 55] / (1 + ((V + p[:, 56]) / (p[:, 57] + eps))**2 + eps)                                                                        
    KQT3_tau_m_s = p[:, 58] + p[:, 59] / (1 + 10**(-p[:, 60] * (p[:, 61] - V)) + eps) + p[:, 62] / (1 + 10**(-p[:, 63] * (p[:, 64] + V)) + eps)                   
    KQT3_w_inf = p[:, 49] + p[:, 50] * xinf_pos((V - p[:, 47])/(p[:, 48] + eps))                                                                 
    KQT3_s_inf = p[:, 53] + p[:, 54] * xinf_pos((V - p[:, 51])/(p[:, 52] + eps))                                                                 
    KQT3_tau_w = p[:, 65] + p[:, 66] / (1 + ((V - p[:, 67]) / (p[:, 68] + eps))**2 + eps)                                                                  
    KQT3_tau_s = p[:, 69]                                                                                                           

    EGL2_m_inf = xinf_neg((V - p[:, 71])/(p[:, 72] + eps))                                                                            
    EGL2_tau_m = p[:, 73] * xinf_pos((V - p[:, 74])/(p[:, 75] + eps)) + p[:, 76]                                                               

    EGL36_m_f_inf = xinf_neg((V - p[:, 78])/(p[:, 79] + eps))                                                                         
    EGL36_m_m_inf = xinf_neg((V - p[:, 78])/(p[:, 79] + eps))                                                                         
    EGL36_m_s_inf = xinf_neg((V - p[:, 78])/(p[:, 79] + eps))                                                                         
    EGL36_tau_m_f = p[:, 82]                                                                                                        
    EGL36_tau_m_m = p[:, 81]                                                                                                        
    EGL36_tau_m_s = p[:, 80]                                                                                                        

    IRK_m_inf = xinf_pos((V - p[:, 84])/(p[:, 85] + eps))                                                                              
    IRK_tau_m = p[:, 86] / (torch.exp(-(V - p[:, 87])/(p[:, 88] + eps)) + torch.exp((V - p[:, 89])/(p[:, 90] + eps)) + eps) + p[:, 91] # 87,88,89,90                                         

    EGL19_m_inf = xinf_neg((V - p[:, 93])/(p[:, 94] + eps))                                                                           
    EGL19_tau_m = (p[:, 103] * torch.exp(-((V - p[:, 104])/(p[:, 105] + eps))**2)) + (p[:, 106] * torch.exp(-((V - p[:, 107])/(p[:, 108] + eps))**2)) + p[:, 109] # 104, 105, 107, 108           
    EGL19_h_inf = (p[:, 99] * xinf_neg((V - p[:, 95])/(p[:, 96] + eps)) + p[:, 100]) * (p[:, 101] * xinf_pos((V - p[:, 97])/(p[:, 98] + eps)) + p[:, 102])      
    EGL19_tau_h = p[:, 110] * (p[:, 111] * xinf_pos((V - p[:, 112])/(p[:, 113] + eps)) + p[:, 114] * xinf_pos((V - p[:, 115])/(p[:, 116] + eps)) + p[:, 117])       

    UNC2_m_inf = xinf_neg((V - p[:, 119])/(p[:, 120] + eps))                                                                          
    UNC2_tau_m = p[:, 123] / (torch.exp(-(V - p[:, 124])/(p[:, 125] + eps)) + torch.exp((V - p[:, 124])/(p[:, 126] + eps))) + p[:, 127] # 124, 125, 126                                  
    UNC2_h_inf = xinf_pos((V - p[:, 121])/(p[:, 122] + eps))                                                                           
    UNC2_tau_h = p[:, 128] * xinf_neg((V - p[:, 129])/(p[:, 130] + eps)) + p[:, 131] * xinf_pos((V - p[:, 132])/(p[:, 133] + eps))                             

    CCA1_m_inf = xinf_neg((V - p[:, 135])/(p[:, 136] + eps))                                                                          
    CCA1_h_inf = xinf_pos((V - p[:, 137])/(p[:, 138] + eps))                                                                           
    CCA1_tau_m = p[:, 139] * xinf_neg((V - p[:, 140])/(p[:, 141] + eps)) + p[:, 142]                                                               
    CCA1_tau_h = p[:, 143] * xinf_pos((V - p[:, 144])/(p[:, 145] + eps)) + p[:, 146]                                                               

    EGL19_alpha = EGL19_m_inf/(EGL19_tau_m + eps)
    EGL19_beta = 1/(EGL19_tau_m + eps) - EGL19_alpha
    UNC2_alpha = UNC2_m_inf/(UNC2_tau_m + eps)
    UNC2_beta = 1/(UNC2_tau_m + eps) - UNC2_alpha

    SLO1_k_neg_o = (p[:, 150] * torch.exp(-p[:, 148] * V)) * (1/(1 + (CA2_pos_n/p[:, 154])**p[:, 155]))
    SLO1_k_neg_c = (p[:, 150] * torch.exp(-p[:, 148] * V)) * (1/(1 + (p[:, 174]/p[:, 154])**p[:, 155]))           
    SLO1_k_pos_o = (p[:, 151] * torch.exp(-p[:, 149] * V)) * (1/(1 + (p[:, 152]/CA2_pos_n)**p[:, 153]))                   

    SLO2_k_neg_o = (p[:, 159] * torch.exp(-p[:, 157] * V)) * (1/(1 + (CA2_pos_n/p[:, 163])**p[:, 164]))
    SLO2_k_neg_c = (p[:, 159] * torch.exp(-p[:, 157] * V)) * (1/(1 + (p[:, 174]/p[:, 163])**p[:, 164]))                                        
    SLO2_k_pos_o = (p[:, 160] * torch.exp(-p[:, 158] * V)) * (1/(1 + (p[:, 161]/CA2_pos_n)**p[:, 162]))                                         

    SLO1_EGL19_m_inf = (EGL19_m_inf * SLO1_k_pos_o * (EGL19_alpha + EGL19_beta + SLO1_k_neg_c)) / (((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + EGL19_alpha) + EGL19_beta * SLO1_k_neg_c) + eps)
    SLO1_EGL19_tau_m = (EGL19_alpha + EGL19_beta + SLO1_k_neg_c) / (((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + EGL19_alpha) + EGL19_beta * SLO1_k_neg_c) + eps)
    SLO1_UNC2_m_inf = (UNC2_m_inf * SLO1_k_pos_o * (UNC2_alpha + UNC2_beta + SLO1_k_neg_c)) / (((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + UNC2_alpha) + UNC2_beta * SLO1_k_neg_c) + eps)
    SLO1_UNC2_tau_m = (UNC2_alpha + UNC2_beta + SLO1_k_neg_c) / (((SLO1_k_pos_o + SLO1_k_neg_o) * (SLO1_k_neg_c + UNC2_alpha) + UNC2_beta * SLO1_k_neg_c) + eps)

    SLO2_EGL19_m_inf = (EGL19_m_inf * SLO2_k_pos_o * (EGL19_alpha + EGL19_beta + SLO2_k_neg_c)) / (((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + EGL19_alpha) + EGL19_beta * SLO2_k_neg_c) + eps)
    SLO2_EGL19_tau_m = (EGL19_alpha + EGL19_beta + SLO2_k_neg_c) / (((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + EGL19_alpha) + EGL19_beta * SLO2_k_neg_c) + eps)
    SLO2_UNC2_m_inf = (UNC2_m_inf * SLO2_k_pos_o * (UNC2_alpha + UNC2_beta + SLO2_k_neg_c)) / (((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + UNC2_alpha) + UNC2_beta * SLO2_k_neg_c) + eps)
    SLO2_UNC2_tau_m = (UNC2_alpha + UNC2_beta + SLO2_k_neg_c) / (((SLO2_k_pos_o + SLO2_k_neg_o) * (SLO2_k_neg_c + UNC2_alpha) + UNC2_beta * SLO2_k_neg_c) + eps)

    KCNL_m_inf = CA2_pos_m / (p[:, 166] + CA2_pos_m + eps)                                                                                
    KCNL_tau_m = p[:, 167]                                                                                                          

    dV = (-(I_K + I_C + I_KCa + I_L) + iext) / p[:, 214]
    Dm_SHL1 = (SHL1_m_inf - m_SHL1) / (SHL1_tau_m + eps)
    Dh_SHL1_f = (SHL1_h_f_inf - h_SHL1_f) / (SHL1_tau_h_f + eps)
    Dh_SHL1_s = (SHL1_h_s_inf - h_SHL1_s) / (SHL1_tau_h_s + eps)
    Dm_KVS1 = (KVS1_m_inf - m_KVS1) / (KVS1_tau_m + eps)
    Dh_KVS1 = (KVS1_h_inf - h_KVS1) / (KVS1_tau_h + eps)
    Dm_SHK1 = (SHK1_m_inf - m_SHK1) / (SHK1_tau_m + eps)
    Dh_SHK1 = (SHK1_h_inf - h_SHK1) / (SHK1_tau_h + eps)
    Dm_KQT3_f = (KQT3_m_f_inf - m_KQT3_f) / (KQT3_tau_m_f + eps)
    Dm_KQT3_s = (KQT3_m_s_inf - m_KQT3_s) / (KQT3_tau_m_s + eps)
    Dw_KQT3 = (KQT3_w_inf - w_KQT3) / (KQT3_tau_w + eps)
    Ds_KQT3 = (KQT3_s_inf - s_KQT3) / (KQT3_tau_s + eps)
    Dm_EGL2 = (EGL2_m_inf - m_EGL2) / (EGL2_tau_m + eps)
    Dm_EGL36_f = (EGL36_m_f_inf - m_EGL36_f) / (EGL36_tau_m_f + eps)
    Dm_EGL36_m = (EGL36_m_m_inf - m_EGL36_m) / (EGL36_tau_m_m + eps)
    Dm_EGL36_s = (EGL36_m_s_inf - m_EGL36_s) / (EGL36_tau_m_s + eps)
    Dm_IRK = (IRK_m_inf - m_IRK) / (IRK_tau_m + eps)
    Dm_EGL19 = (EGL19_m_inf - m_EGL19) / (EGL19_tau_m + eps)
    Dh_EGL19 = (EGL19_h_inf - h_EGL19) / (EGL19_tau_h + eps)
    Dm_UNC2 = (UNC2_m_inf - m_UNC2) / (UNC2_tau_m + eps)
    Dh_UNC2 = (UNC2_h_inf - h_UNC2) / (UNC2_tau_h + eps)
    Dm_CCA1 = (CCA1_m_inf - m_CCA1) / (CCA1_tau_m + eps)
    Dh_CCA1 = (CCA1_h_inf - h_CCA1) / (CCA1_tau_h + eps)
    Dm_SLO1_EGL19 = (SLO1_EGL19_m_inf - m_SLO1_EGL19) / (SLO1_EGL19_tau_m + eps)#
    Dm_SLO1_UNC2 = (SLO1_UNC2_m_inf - m_SLO1_UNC2) / (SLO1_UNC2_tau_m + eps)#
    Dm_SLO2_EGL19 = (SLO2_EGL19_m_inf - m_SLO2_EGL19) / (SLO2_EGL19_tau_m + eps)#
    Dm_SLO2_UNC2 = (SLO2_UNC2_m_inf - m_SLO2_UNC2) / (SLO2_UNC2_tau_m + eps)#
    Dm_KCNL = (KCNL_m_inf - m_KCNL) / (KCNL_tau_m + eps)
    DCA2_pos_m = DCA2_pos_m_1 + DCA2_pos_m_2#

    dV_m = torch.cat([dV, Dm_SHL1, Dh_SHL1_f, Dh_SHL1_s, Dm_KVS1, Dh_KVS1, Dm_SHK1, Dh_SHK1, Dm_KQT3_f, Dm_KQT3_s, Dw_KQT3, Ds_KQT3, Dm_EGL2,
        Dm_EGL36_f, Dm_EGL36_m, Dm_EGL36_s, Dm_IRK, Dm_EGL19, Dh_EGL19, Dm_UNC2, Dh_UNC2, Dm_CCA1, Dh_CCA1, Dm_SLO1_EGL19, Dm_SLO1_UNC2, Dm_SLO2_EGL19, Dm_SLO2_UNC2, Dm_KCNL, DCA2_pos_m], dim = 2)

    return dV_m # (128, 500, 319)