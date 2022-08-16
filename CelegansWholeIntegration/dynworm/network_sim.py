
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
from diffeqpy import de, ode 

np.random.seed(10)

# TODO: REAL-TIME PLOTTING OF NEURAL VOLTAGE FOR SELECTED NEURON

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


########################################################################################################################################################################
### MASTER EXECUTION FUNCTIONS (BASIC) #################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

def run_network_constinput(t_duration, input_vec, ablation_mask, \
    custom_initcond = False, ablation_type = "all", verbose = True):

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

    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

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
    interp_kind_input = 'nearest', verbose = True):

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

    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

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
    interp_kind_input = 'nearest', interp_kind_voltage = 'linear', verbose = True):
    
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

    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

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
    custom_muscle_map = False, fdb_init = 1.38, t_delay = 0.54, reaction_scaling = 1, verbose = True):

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

    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

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
    custom_muscle_map = False, fdb_init = 1.38, t_delay = 0.54, reaction_scaling = 1, verbose = True):

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

    if params_obj_neural['nonlinear_AWA'] + params_obj_neural['nonlinear_AVL'] == 0:

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

    """ leak current """

    LCurr = params_obj_neural['AWA_nonlinear_params']['gL']*(v - params_obj_neural['AWA_nonlinear_params']['vL'])

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
    
    KCurr = (params_obj_neural['AWA_nonlinear_params']['gK']*w+params_obj_neural['AWA_nonlinear_params']['gK7']*\
        slo2+params_obj_neural['AWA_nonlinear_params']['gK4']*slo+params_obj_neural['AWA_nonlinear_params']['gK6']+params_obj_neural['AWA_nonlinear_params']['gK3']*\
        yinf*(1-bk)+params_obj_neural['AWA_nonlinear_params']['gK5']*kb)*(v-params_obj_neural['AWA_nonlinear_params']['vK']) + params_obj_neural['AWA_nonlinear_params']['gK2']*kir

    """ calcium current """

    CaCurr = params_obj_neural['AWA_nonlinear_params']['gCa']*(c1+params_obj_neural['AWA_nonlinear_params']['fac']*c2)*(v-params_obj_neural['AWA_nonlinear_params']['vCa'])

    dv_AWA_vec = np.zeros(params_obj_neural['N'])
    dw_AWA_vec = np.zeros(params_obj_neural['N'])
    dc1_AWA_vec = np.zeros(params_obj_neural['N'])
    dc2_AWA_vec = np.zeros(params_obj_neural['N'])
    dbk_AWA_vec = np.zeros(params_obj_neural['N'])
    dslo_AWA_vec = np.zeros(params_obj_neural['N'])
    dslo2_AWA_vec = np.zeros(params_obj_neural['N'])
    dkb_AWA_vec = np.zeros(params_obj_neural['N']) 

    np.put(dv_AWA_vec, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], LCurr + KCurr + CaCurr)
    np.put(dw_AWA_vec, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], (xinf-w)/params_obj_neural['AWA_nonlinear_params']['TK'])
    np.put(dc1_AWA_vec, params_obj_neural['AWA_nonlinear_params']['AWA_inds'], (minf*winf/params_obj_neural['AWA_nonlinear_params']['mx']-c1)/params_obj_neural['AWA_nonlinear_params']['TC1']-\
        minf*winf*c2/(params_obj_neural['AWA_nonlinear_params']['mx']*params_obj_neural['AWA_nonlinear_params']['TC1'])-c1/(2*params_obj_neural['AWA_nonlinear_params']['TC2']*tau)+c2/\
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

    else:

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