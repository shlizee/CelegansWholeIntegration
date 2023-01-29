
# coding: utf-8

########################################################################################################################################################################
# UTILITY FUNCTIONS ####################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

import os

import json
import numpy as np
import scipy.stats as spstats
import h5py
from scipy import signal, interpolate
from statsmodels.tsa.api import ExponentialSmoothing
from itertools import combinations, chain
from scipy.special import comb

from scipy.ndimage.filters import gaussian_filter
from dynworm import sys_paths as paths
from dynworm import neural_params as n_params
from dynworm import neurons_idx as n_idx

import matplotlib.pyplot as plt

def load_Json(filename):

    with open(filename) as content:

        content = json.load(content)

    return content

def construct_dyn_inputmat(t0, tf, input_type, neuron_indices, normalized_amps = False, freqs = False, noise_amplitudes = False, step_time_interval = False):

    dt = n_params.pA_unit_baseline['dt']/n_params.pA_unit_baseline['time_scaler']

    timepoints = np.arange(t0, tf, dt)
    input_mat = np.zeros((len(timepoints) + 1, n_params.pA_unit_baseline['N']))

    if input_type == 'sinusoidal':
        
        amps = np.asarray(normalized_amps) / 2.

        for i in range(len(neuron_indices)):

            for j in range(len(timepoints)):
                
                input_mat[j, neuron_indices[i]] = amps[i] * np.sin(freqs[i] * timepoints[j]) + amps[i]

    elif input_type == 'noisy':

        for k in range(len(neuron_indices)):

            noise = 10**(-2)*np.random.normal(0, noise_amplitudes[k], len(input_mat))
            input_mat[:, neuron_indices[k]] = normalized_amps[k] + noise

    return input_mat

def redblue(m):

    m1 = m * 0.5
    r = np.divide(np.arange(0, m1)[:, np.newaxis], np.max([m1-1,1]))
    g = r
    r = np.vstack([r, np.ones((int(m1), 1))])
    g = np.vstack([g, np.flipud(g)])
    b = np.flipud(r)
    x = np.linspace(0, 1, m)[:, np.newaxis]

    red = np.hstack([x, r, r])
    green = np.hstack([x, g, g])
    blue = np.hstack([x, b, b])

    red_tuple = tuple(map(tuple, red))
    green_tuple = tuple(map(tuple, green))
    blue_tuple = tuple(map(tuple, blue))

    cdict = {
    	'red': red_tuple,
        'green': green_tuple,
        'blue': blue_tuple
        }

    return cdict

def project_v_onto_u(v, u):
    
    factor = np.divide(np.dot(u, v), np.power(np.linalg.norm(u), 2))
    projected = factor * u
    
    return projected

def compute_mean_velocity_manual(x, y, directional_ind_1, directional_ind_2, body_ind, dt, scaling_factor):
    
    # directional vectors
    
    x_pos_components = x[:, directional_ind_1] - x[:, directional_ind_2]
    y_pos_components = y[:, directional_ind_1] - y[:, directional_ind_2]
    positional_vecs = np.vstack([x_pos_components, y_pos_components])[:, :-1]
    
    # velocity vectors using central difference
    
    #x_vel_components = np.diff(x[:, body_ind])
    #y_vel_components = np.diff(y[:, body_ind])
    #velocity_vecs = np.vstack([x_vel_components, y_vel_components])
    
    computed_vels = np.zeros(len(positional_vecs[0, :]))
    computed_signs = np.zeros(len(positional_vecs[0, :]))    

    for k in range(len(positional_vecs[0, :]))[1:]:
        
        left_term_x = x[k-1, body_ind]
        right_term_x = x[k+1, body_ind]
        left_term_y = y[k-1, body_ind]
        right_term_y = y[k+1, body_ind]
        
        velocity = np.array([right_term_x - left_term_x, right_term_y - left_term_y]) / (scaling_factor*dt)
        #print(velocity)
        
        projected = project_v_onto_u(velocity, positional_vecs[:, k])
        #projected_vel_norms[k] = np.linalg.norm(projected) * np.sign(np.dot(projected, positional_vecs[:, k]))
        sign = np.sign(np.dot(projected, positional_vecs[:, k]))
        computed_vels[k] = sign * np.linalg.norm(velocity)
        computed_signs[k] = sign
        
    mean_velocity = np.mean(computed_vels)
    
    return computed_vels, computed_signs, mean_velocity

def compute_mean_velocity(x, y, directional_ind_1, directional_ind_2, body_ind, dt, scaling_factor):
    
    # directional vectors
    
    x_pos_components = x[:, directional_ind_1] - x[:, directional_ind_2]
    y_pos_components = y[:, directional_ind_1] - y[:, directional_ind_2]
    positional_vecs = np.vstack([x_pos_components, y_pos_components])[:, :-1]
    
    # velocity vectors using central difference
    
    x_vel_components = np.gradient(x[:, body_ind], edge_order = 2) * (scaling_factor/dt)
    y_vel_components = np.gradient(y[:, body_ind], edge_order = 2) * (scaling_factor/dt)
    velocity_vecs = np.vstack([x_vel_components, y_vel_components])
    
    computed_vels = np.zeros(len(positional_vecs[0, :]))
    computed_signs = np.zeros(len(positional_vecs[0, :]))        

    for k in range(len(positional_vecs[0, :])):
        
        projected = project_v_onto_u(velocity_vecs[:, k], positional_vecs[:, k])
        sign = np.sign(np.dot(projected, positional_vecs[:, k]))
        computed_vels[k] = sign * np.linalg.norm(velocity_vecs[:, k])
        computed_signs[k] = sign
        
    mean_velocity = np.mean(computed_vels)
    
    return computed_vels, computed_signs, mean_velocity

def mean_confidence_interval(data, confidence=0.99):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), spstats.sem(a)
    h = se * spstats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h, h

def compute_chemotaxis_index(inmask_mat, neuron_ind, ref_signal):
    
    stim_integral = np.sum(inmask_mat[:, neuron_ind])
    CI = (stim_integral - ref_signal) / ref_signal
    
    return CI

def continuous_transition_scaler(old, new, t, rate, tSwitch):

    return np.multiply(old, 0.5-0.5*np.tanh((t-tSwitch)/rate)) + np.multiply(new, 0.5+0.5*np.tanh((t-tSwitch)/rate))

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N

def compute_segment_angles(x, y):
    
    phi_segments = []

    for k in range(len(x)):

        #segment_vecs = np.vstack([np.diff(x[k][::4]), np.diff(y[k][::4])]).T
        segment_vecs = np.vstack([np.diff(x[k][::2]), np.diff(y[k][::2])]).T
        
        angles_list = []
        
        for j in range(len(segment_vecs) - 1):

            v0 = segment_vecs[j]
            v1 = segment_vecs[j+1]

            angles_list.append(np.degrees(np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))))
        
        angles_vec = np.asarray(angles_list)
        phi_segments.append(angles_vec)
        
        #print(k)
        
    return np.vstack(phi_segments)

def save_dict(filename, dict_object):

    np.savez(filename, **dict_object)

def compute_eigworm_coeffs_alternative(phi, eigenworm_modes):

    phi_truncated = phi[200:]

    interpolate_phi = interpolate.interp1d(np.arange(0, 24), phi_truncated, axis=1)
    expanded_phi = interpolate_phi(np.linspace(0, 23, 48)) 
    expanded_phi_normalized = np.subtract(expanded_phi, np.tile(np.mean(expanded_phi, axis = 1), (48, 1)).T)

    M_dot_V = np.dot(expanded_phi_normalized, eigenworm_modes)

    proj1 = M_dot_V[:, 0]
    proj2 = M_dot_V[:, 1]
    proj3 = M_dot_V[:, 2]
    proj4 = M_dot_V[:, 3]
    proj5 = M_dot_V[:, 4]
    proj6 = M_dot_V[:, 5]
    proj7 = M_dot_V[:, 6]

    s1 = np.linalg.norm(proj1)**2
    s2 = np.linalg.norm(proj2)**2
    s3 = np.linalg.norm(proj3)**2
    s4 = np.linalg.norm(proj4)**2
    s5 = np.linalg.norm(proj5)**2
    s6 = np.linalg.norm(proj6)**2
    s7 = np.linalg.norm(proj7)**2

    s_list = [s1,s2,s3,s4,s5,s6,s7]
    # square before taking the cumulataive sum
    s_list_normalized = s_list / np.sum(s_list)
    s_list_normalized_cumulative = np.cumsum(s_list_normalized)

    return s_list_normalized_cumulative

def compute_eigworm_CI(eig_coeff_stacked, confidence):

    mean_coeffcients = []
    ci_list = []

    for k in range(6):

        mean, low, high, h = mean_confidence_interval(eig_coeff_stacked[:, k], confidence = confidence)
        mean_coeffcients.append(mean)
        ci_list.append(h)
        
    return mean_coeffcients, ci_list

def compute_CI_from_swarm(stored_x, stored_y, center_coords, boundary_radius, batchsize):
    
    # Take the last points
    
    x_samples = []
    y_samples = []

    for k in range(len(stored_x)):

        x_samples.append(stored_x[k][-1][0])
        y_samples.append(stored_y[k][-1][0])

    x_samples = np.asarray(x_samples)
    y_samples = np.asarray(y_samples)
    
    # randomize
    
    indexes = np.arange(len(stored_x))
    np.random.shuffle(indexes)
    x_samples_shuffled = x_samples[indexes]
    y_samples_shuffled = y_samples[indexes]
    
    # Normalize around center of the gradient
    
    x_samples_normalized = np.subtract(x_samples_shuffled, np.ones(len(stored_x)) * center_coords[0])
    y_samples_normalized = np.subtract(y_samples_shuffled, np.ones(len(stored_y)) * center_coords[1])
    
    # Split into batches
    
    x_samples_split = np.split(x_samples_normalized, batchsize)
    y_samples_split = np.split(y_samples_normalized, batchsize)
    
    # Compute CI for each batch
    
    CI_list = []
    
    for batch_num in range(len(x_samples_split)):
        
        radiuses = np.sqrt(np.add(np.power(x_samples_split[batch_num], 2), np.power(y_samples_split[batch_num],2)))
        within_test_region = radiuses < boundary_radius
        print(within_test_region)
        num_inside = np.sum(within_test_region)
        num_outside = len(within_test_region) - num_inside
        CI = (num_inside - num_outside) / (num_inside + num_outside)
        
        CI_list.append(CI)
    
    return CI_list

def alpha_func(b, window_size, dt):

    time = np.arange(window_size) * dt

    alpha_func = np.exp(b*time)
    alpha_func = alpha_func / np.max(alpha_func)

    alpha_func = np.multiply(time, alpha_func)

    return alpha_func    

def convolve_alpha_func(a, v_sample, kernel):
    
    #kernel = np.flip(kernel)
    
    v_sample_scaled = a * v_sample
    
    filtered = signal.convolve(v_sample_scaled, kernel, mode='valid') / sum(kernel)
    filtered = np.pad(filtered, (len(kernel)//2, len(kernel)//2 - 1), 'constant', constant_values=(filtered[0], filtered[-1]))
    
    return filtered

def convolve_single_exponential(time_constant, dt, cystoplasmic_data):
    
    alpha = 1 - np.exp((-dt)/time_constant)
    
    exp = ExponentialSmoothing(cystoplasmic_data)
    exp_model = exp.fit(smoothing_level=alpha)
    result = exp_model.fittedvalues
    
    return result

def voltage_2_calcium(v_sample, a, b, window_size, dt_sim, dt_exp, time_constant):
    
    alpha_kernel = alpha_func(b, window_size, dt_sim)
    voltage_2_cystoplasmic = convolve_alpha_func(a, v_sample, alpha_kernel)
    cystoplasmic_2_nuclear = convolve_single_exponential(time_constant, dt_exp, voltage_2_cystoplasmic)
    
    output = {'kernel': alpha_kernel,
              'cystoplasmic': voltage_2_cystoplasmic,
              'nuclear': cystoplasmic_2_nuclear}
    
    return output

def find_neighbor_interneurons(neuron_inds):

    # Considered circuit - Postsynaptic neurons + gap connected neurons + Kim et al circuit

    neighbors = []

    for neuron_ind in neuron_inds:

        for neuron in np.where(n_params.Gs_Static[:, neuron_ind])[0]:
            
            if n_idx.neurons_list[neuron]['group'] == 'inter':
            
                neighbors.append(neuron)
                print(n_idx.neuron_names[neuron] + ' syn')

        for neuron in np.where(n_params.Gg_Static[:, neuron_ind])[0]:

            if n_idx.neurons_list[neuron]['group'] == 'inter':
            
                neighbors.append(neuron)
                print(n_idx.neuron_names[neuron] + ' gap')

    neighbors = np.asarray(neighbors)
    neighbors = np.unique(neighbors)
            
    return neighbors

def neuron_inds_2_names(inds):
    
    names = []
    
    for ind in inds:
        
        names.append(n_idx.neuron_names[ind])
        
    names = np.asarray(names)
    
    return names

def neuron_names_2_inds(names):
    
    inds = []
    
    for name in names:
        
        inds.append(np.where(np.asarray(n_idx.neuron_names) == name)[0])
        
    inds = np.asarray(inds).flatten()
    
    return inds

def make_smooth_pulse(time_length, dt, pre_stim, post_stim, rate):
    
    time_discretized_arr = np.arange(0, time_length, dt)
    current_arr = np.zeros(time_discretized_arr.shape)
    
    k = 0
    
    for k in range(len(time_discretized_arr)):
        
        tk = time_discretized_arr[k]
        
        if tk < time_length // 2:
            
            current_arr[k] = continuous_transition_scaler(pre_stim, post_stim, tk, rate, time_length / 10)
            
        else:
            
            current_arr[k] = continuous_transition_scaler(post_stim, pre_stim, tk, rate, (time_length / 10) * 9)
            
    return time_discretized_arr, current_arr

def all_possible_combinations(candidates):
    
    combination_list = []
    
    for k in range(1, len(candidates)):
        
        c = list(combinations(candidates, k))
        
        for row in np.array(c):
            
            combination_list.append(row)
    
    return combination_list

def generate_pulse_inputs(duration, amp, freq, plot = False):
    
    input_mat = np.zeros((duration)) # 1 = 10ms
    t = np.linspace(0, freq, duration - round(duration/(freq*2)), endpoint=False)

    pulse = (amp * signal.square(2 * np.pi * t) + amp)/2

    input_mat[:round(duration/(freq*2))] = 0
    input_mat[round(duration/(freq*2)):] = pulse

    if plot:
    
        plt.plot(input_mat[:])
        plt.title('input mat')
        plt.ylabel('amplitude (pA)')
        plt.xlabel('time (10ms)')
        plt.show()
    
    return input_mat

def swarm_init(num):
    
    x_init = np.array([0.0]*num)
    y_init = np.array([0.0]*num)
    orientation = np.array([0.0]*num)
    angle_step = 360/num
    
    for i in range(num):
        orientation[i] = i*angle_step
        x_init[i] = math.cos(math.radians(orientation[i]))*125
        y_init[i] = math.sin(math.radians(orientation[i]))*125
    
    return x_init, y_init, orientation