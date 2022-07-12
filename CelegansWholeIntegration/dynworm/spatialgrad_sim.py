# coding: utf-8

###########################################################################################################################################################################
# VISCOELASTIC ROD MODEL SIMULATION WITH SPATIAL GRADIENT #################################################################################################################
# TO BE RELEASED IN BLOCK 2 VERSION OF THE API ############################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################

########################################################################################################################################################################

# TODO: SYNCHRONIZE NETWORK SIM TIMERANGE WITH BODY SIM

import os

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import integrate, sparse, linalg, interpolate

from dynworm import neural_params as n_params
from dynworm import body_params as b_params
from dynworm import neurons_idx as n_idx
from dynworm import network_sim as n_sim
from dynworm import body_sim as b_sim
from dynworm import sys_paths as paths
from dynworm import utils

nonlinear_voltage_renorm = np.load('saved_data/bargmann_nonlinear/nonlinear_V_adjustments.npy')

np.random.seed(10)

def solve_body_spatialgrads(input_grid_x, input_grid_y, input_grid_mat, input_neurons_body_locs, ext_voltage_mat, ablation_mask, t_delta, \
    input_neurons_dependent_mat = False, sense_derivative = False, derivative_scaling = 10, with_feedback = False, \
    custom_initcond = False, custom_muscle_map = False, ablation_type = "all", \
    xinit = 0, yinit = 0, orientation_angle = 0, \
    fdb_init = 1.36, t_delay = 0.5, reaction_scaling = 1, \
    kappa_scaling = 1, kappa_forcing = False, noise_amplitude = 0, ambient_noise = False, negative_stim = False):

    ########################################################################################################################################################################
    # Neural integration prep ##############################################################################################################################################
    ########################################################################################################################################################################

    # input_neurons_body_locs = (279, 24)
    # input_neurons_dependent_mat = (279, 279)

    """ define the time range to be simulated """

    t0 = 0
    tf = (len(ext_voltage_mat) - 1) * t_delta 
    dt = t_delta

    nsteps_traj_neural = len(ext_voltage_mat) 
    nsteps_traj_body = nsteps_traj_neural - 100 + 1 # Figure out out of bound issue

    timepoints_traj_neural = np.linspace(t0, tf, nsteps_traj_neural)
    timepoints_traj_force =  np.linspace(t0, tf - 1 + dt, nsteps_traj_body)

    """ Tracking metric """

    progress_milestones = np.linspace(0, nsteps_traj_neural, 10).astype('int')

    n_sim.params_obj_neural['simulation_type'] = 'spatial_gradient'

    n_sim.params_obj_neural['interpolate_voltage'] = interpolate.interp1d(timepoints_traj_neural, ext_voltage_mat, axis=0, kind = 'linear', fill_value = "extrapolate")
    n_sim.params_obj_neural['fdb_init'] = fdb_init
    n_sim.params_obj_neural['t_delay'] = t_delay
    n_sim.params_obj_neural['reaction_scaling'] = reaction_scaling

    """ define the connectivity """

    n_sim.modify_Connectome(ablation_mask, ablation_type)

    """ Define initial condition """

    if type(custom_initcond) == bool:

        initcond_neural = 10**(-4)*np.random.normal(0, 0.94, 2*n_sim.params_obj_neural['N'])

    else:

        initcond_neural = custom_initcond
        print("using the custom initial condition")

    """ Define muscle map for feedback """

    n_sim.params_obj_neural['muscle_map'] = b_params.muscle_map_f
    n_sim.params_obj_neural['muscle_map_pseudoinv'] = b_params.muscle_map_pseudoinv_f

    print("Network integration prep completed...")

    ########################################################################################################################################################################
    # Body integration prep ################################################################################################################################################
    ########################################################################################################################################################################

    if custom_muscle_map == False:

        b_sim.params_obj_body['muscle_map'] = b_params.muscle_map_f

    else:

        b_sim.params_obj_body['muscle_map'] = custom_muscle_map
        print("using custom muscle map")

    b_sim.params_obj_body['kappa_scaling'] = kappa_scaling

    if type(kappa_forcing) != bool:

        b_sim.params_obj_body['kappa_forcing'] = kappa_forcing

    else:

        b_sim.params_obj_body['kappa_forcing'] = np.zeros(23)
    
    b_sim.compute_initcond(b_sim.params_obj_body, xinit, yinit, orientation_angle)
    b_sim.compute_params_spatial_grad(b_sim.params_obj_body, nsteps_traj_body)

    initcond_body = b_sim.params_obj_body['IC']
    N_body = len(initcond_body)

    b_sim.params_obj_body['interpolate_grid'] = interpolate.interp2d(input_grid_x, input_grid_y, input_grid_mat, kind = 'cubic')
    print("Body integration prep completed...")

    ########################################################################################################################################################################
    # Configuring the ODE Solver ###########################################################################################################################################
    ########################################################################################################################################################################

    if with_feedback != True:

        r_neural = integrate.ode(n_sim.membrane_voltageRHS_vext_spatialgrad, n_sim.compute_jacobian_vext_spatialgrad).set_integrator('vode', atol = 1e-3, method = 'bdf')

    else:

        r_neural = integrate.ode(n_sim.membrane_voltageRHS_fdb_vext_spatialgrad, n_sim.compute_jacobian_fdb_vext_spatialgrad).set_integrator('vode', atol = 1e-3, method = 'bdf')

    r_neural.set_initial_value(initcond_neural, t0)

    t = np.zeros(nsteps_traj_neural)
    traj_neural = np.zeros((nsteps_traj_neural, n_sim.params_obj_neural['N']))
    s_traj = np.zeros((nsteps_traj_neural, n_sim.params_obj_neural['N']))
    vthmat = np.zeros((nsteps_traj_neural, n_sim.params_obj_neural['N']))

    n_sim.params_obj_neural['interpolate_traj'] = interpolate.interp1d(timepoints_traj_neural, traj_neural, axis=0)
    n_sim.params_obj_neural['interpolate_vthmat'] = interpolate.interp1d(timepoints_traj_neural, vthmat, axis=0)

    traj_body = np.zeros((nsteps_traj_body, N_body))
    traj_body[0, :] = initcond_body[:N_body]

    x_whole_init, y_whole_init = b_sim.solve_xy_individual(traj_body[0,0], traj_body[0,1], traj_body[0, 2:26])

    inmask_body = infer_input(input_neurons_body_locs, x_whole_init, y_whole_init, noise_amplitude, ambient_noise, negative_stim)

    if sense_derivative == True:
        n_sim.params_obj_neural['inmask'] = np.zeros(n_sim.params_obj_neural['N'])

    else:
        n_sim.params_obj_neural['inmask'] = inmask_body

    if type(input_neurons_dependent_mat) != bool:

        dependent_input_mat = np.multiply(input_neurons_dependent_mat, np.tile(n_sim.params_obj_neural['inmask'], (279, 1)))
        dependent_input_vec = dependent_input_mat.sum(axis = 1)

        n_sim.params_obj_neural['inmask'] = n_sim.params_obj_neural['inmask'] + dependent_input_vec

    t[0] = t0
    traj_neural[0, :] = initcond_neural[:n_sim.params_obj_neural['N']]
    s_traj[0, :] = initcond_neural[n_sim.params_obj_neural['N']:]
    vthmat[0, :] = n_sim.EffVth_rhs(n_sim.params_obj_neural['inmask'])

    inmask_list = []
    inmask_list_derivative = []
    inmask_list.append(inmask_body)
    inmask_list_derivative.append(n_sim.params_obj_neural['inmask'])
    print("Configured Integration solver...")

    ########################################################################################################################################################################
    # Integrate both neural and body ODE(s) simultaneously  ################################################################################################################
    ########################################################################################################################################################################

    k = 1

    while r_neural.successful() and k < nsteps_traj_neural:

        r_neural.integrate(r_neural.t + dt)
        t[k] = r_neural.t
        traj_neural[k, :] = r_neural.y[:n_sim.params_obj_neural['N']]
        s_traj[k, :] = r_neural.y[n_sim.params_obj_neural['N']:]
        vthmat[k, :] = n_sim.EffVth_rhs(n_sim.params_obj_neural['inmask'])
        v_sol_k = np.subtract(traj_neural[k, :n_sim.params_obj_neural['N']], vthmat[k, :])
        v_sol_k = n_sim.voltage_filter(v_sol_k, 200, 1)

        n_sim.params_obj_neural['interpolate_traj'] = interpolate.interp1d(timepoints_traj_neural, traj_neural, axis=0)
        n_sim.params_obj_neural['interpolate_vthmat'] = interpolate.interp1d(timepoints_traj_neural, vthmat, axis=0)

        if k < 100: # Stabilization stage

            inmask_list.append(inmask_list[-1])
            inmask_list_derivative.append(n_sim.params_obj_neural['inmask'])

        elif k == 100: # Worm starts sensing the stimuli from here

            b_sim.neuron_voltages_2_muscles_instant(v_sol_k[:n_sim.params_obj_neural['N']], k-100)
            b_sim.neuron_voltages_2_muscles_instant(v_sol_k[:n_sim.params_obj_neural['N']], k-99)
            b_sim.params_obj_body['interpolate_force'] = interpolate.interp1d(timepoints_traj_force, b_sim.params_obj_body['force'], axis = 0, kind = 'nearest', fill_value = "extrapolate")

            t_interval = np.asarray([0, dt])
            traj_body[k-99, :] = integrate.solve_ivp(b_sim.visco_elastic_rod_rhs, t_interval, traj_body[0, :], method = 'RK45')['y'][:, -1]

            x_nose_k = traj_body[k-99, 0]
            y_nose_k = traj_body[k-99, 1]
            x_whole_k, y_whole_k = b_sim.solve_xy_individual(x_nose_k, y_nose_k, traj_body[k-99, 2:26])

            inmask_body = infer_input(input_neurons_body_locs, x_whole_k, y_whole_k, noise_amplitude, ambient_noise, negative_stim)

            if sense_derivative == True:
                inmask_new = inmask_body.copy()
                inmask_old = inmask_list[-1].copy()
                n_sim.params_obj_neural['inmask'] = -np.divide(np.subtract(inmask_new, inmask_old), t_delta) * derivative_scaling

            else:
                inmask_new = inmask_body.copy()
                n_sim.params_obj_neural['inmask'] = inmask_new.copy()

            if type(input_neurons_dependent_mat) != bool:

                dependent_input_mat = np.multiply(input_neurons_dependent_mat, np.tile(n_sim.params_obj_neural['inmask'], (279, 1)))
                dependent_input_vec = dependent_input_mat.sum(axis = 1)

                n_sim.params_obj_neural['inmask'] = n_sim.params_obj_neural['inmask'] + dependent_input_vec

            inmask_list.append(inmask_new)
            inmask_list_derivative.append(n_sim.params_obj_neural['inmask'])

        elif k > 100:

            force_time_ref_0 = (k-100) * dt
            force_time_ref_1 = force_time_ref_0 + dt

            b_sim.neuron_voltages_2_muscles_instant(v_sol_k[:n_sim.params_obj_neural['N']], k - 99)
            
            b_sim.params_obj_body['interpolate_force'] = interpolate.interp1d(timepoints_traj_force, b_sim.params_obj_body['force'], axis = 0, kind = 'nearest', fill_value = "extrapolate")

            t_interval = np.asarray([force_time_ref_0, force_time_ref_1])
            traj_body[k-99, :] = integrate.solve_ivp(b_sim.visco_elastic_rod_rhs, t_interval, traj_body[k-100, :], method = 'RK45')['y'][:, -1]

            x_nose_k = traj_body[k-99, 0]
            y_nose_k = traj_body[k-99, 1]
            x_whole_k, y_whole_k = b_sim.solve_xy_individual(x_nose_k, y_nose_k, traj_body[k-99, 2:26])

            inmask_body = infer_input(input_neurons_body_locs, x_whole_k, y_whole_k, noise_amplitude, ambient_noise, negative_stim)

            if sense_derivative == True:
                inmask_new = inmask_body.copy()
                inmask_old = inmask_list[-1].copy()
                n_sim.params_obj_neural['inmask'] = -np.divide(np.subtract(inmask_new, inmask_old), t_delta) * derivative_scaling

            else:
                inmask_new = inmask_body.copy()
                n_sim.params_obj_neural['inmask'] = inmask_new.copy()

            if type(input_neurons_dependent_mat) != bool:

                dependent_input_mat = np.multiply(input_neurons_dependent_mat, np.tile(n_sim.params_obj_neural['inmask'], (279, 1)))
                dependent_input_vec = dependent_input_mat.sum(axis = 1)

                n_sim.params_obj_neural['inmask'] = n_sim.params_obj_neural['inmask'] + dependent_input_vec

            inmask_list.append(inmask_new)
            inmask_list_derivative.append(n_sim.params_obj_neural['inmask'])
        
        k += 1

        if k in progress_milestones:

            print(str(np.round((float(k) / nsteps_traj_neural) * 100, 1)) + '% ' + 'completed')

    x, y = b_sim.solve_xy(traj_body[:, 0], traj_body[:, 1], traj_body[:, 2:26])
    
    print("Post-processing the simulated body...")
    x_post, y_post = b_sim.postprocess_xy(x, y)

    result_dict_body = {
        "t" : t,
        "x" : x_post,
        "y" : y_post,
        "phi" : traj_body[:, 2:26],
        "raw_v_solution" : traj_neural[:, :n_sim.params_obj_neural['N']],
        "s_solution": s_traj,
        "vthmat" : vthmat,
        "v_solution" : n_sim.voltage_filter(np.subtract(traj_neural[:, :n_sim.params_obj_neural['N']], vthmat), 200, 1),
        "force" : b_sim.params_obj_body['force'],
        "inmask_mat" : np.vstack(inmask_list),
        "inmask_mat_derivative" : np.vstack(inmask_list_derivative)
        }

    return result_dict_body

########################################################################################################################################################################
# SIMULATION SPECIFIC ENVIRONMENT FUNCTIONS ############################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
######################################################################################################################################################################## 

def construct_input_grid(input_grid_x, input_grid_y, sigma_x, sigma_y, mu_x, mu_y, scaler, type = 'gaussian'):

    if type == 'gaussian':

        x_meshgrid, y_meshgrid = np.meshgrid(input_grid_x, input_grid_y)

        x_term = np.divide(np.power(x_meshgrid - mu_x, 2), 2 * sigma_x**2)
        y_term = np.divide(np.power(y_meshgrid - mu_y, 2), 2 * sigma_y**2)

        xy_term = -(x_term + y_term)
        f_xy = np.exp(xy_term) * scaler

    elif type == 'constant':

        f_xy = np.ones((len(input_grid_y), len(input_grid_x))) * scaler

    return f_xy

def construct_input_grid_circular(input_grid_x, input_grid_y, x0, y0, radius):

    x_meshgrid, y_meshgrid = np.meshgrid(input_grid_x, input_grid_y)

    r = np.sqrt((x_meshgrid - x0)**2 + (y_meshgrid - y0)**2)

    inside_circle = r < radius

    input_grid_zeros = np.zeros((len(input_grid_x), len(input_grid_y)))

    input_grid_zeros[inside_circle] = 1

    return input_grid_zeros

def construct_input_grid_circular_ramp(input_grid_x, input_grid_y, x0, y0, radius_1, radius_2, peak_amplitude):

    radius_diff = (radius_1 - radius_2)

    multiplier = peak_amplitude / radius_diff
    c_slice = construct_input_grid_circular(input_grid_x, input_grid_y, x0, y0, radius_1 - (radius_1/peak_amplitude)*0) * multiplier

    for k in range(1, radius_diff):
        
        c_slice += construct_input_grid_circular(input_grid_x, input_grid_y, x0, y0, radius_1 - (radius_1/peak_amplitude)*k) * multiplier

    return c_slice

def construct_input_grid_perlin_noise(shape, res):
    
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3
    
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:,0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1,1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:,1:].repeat(d[0], 0).repeat(d[1], 1)
    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0], grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = f(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11

    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

def infer_input(input_neurons_body_locs, x_list, y_list, noise_amplitude, ambient_noise, negative_stim):

    input_stim = []

    for k in range(len(x_list)):

        input_stim.append(b_sim.params_obj_body['interpolate_grid'](x_list[k], y_list[k])[0])

    input_stim = np.asarray(input_stim)
    input_stim_mat = np.multiply(input_neurons_body_locs, np.tile(input_stim, (279, 1)))

    input_vec = input_stim_mat.sum(axis = 1)

    if ambient_noise == True:

        noise = np.random.normal(0, noise_amplitude, len(n_idx.sensory_group))
        noise_added = input_vec[n_idx.sensory_group] + noise
        input_vec[n_idx.sensory_group] = noise_added

    else:

        noise = np.random.normal(0, noise_amplitude, np.sum(input_vec > 1))
        noise_added = input_vec[input_vec > 1] + noise
        input_vec[input_vec > 1] = noise_added

        if negative_stim == True:

            input_vec = input_vec * -1

    return input_vec