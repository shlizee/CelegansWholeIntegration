
# coding: utf-8

########################################################################################################################################################################
# PLOTTING FUNCTIONS ###################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from dynworm import sys_paths as path

def plot_colormap(result_dict, sub_neurons = 'all', vmin = -50, vmax = 50, cmap = 'seismic', figsize_w = 15, figsize_h = 10):

    nsteps = result_dict['steps']
    VsubVth = result_dict['v_solution'].transpose()

    #os.chdir(path.data_4_analysis_dir)

    """ Sub Neurons """

    if sub_neurons == 'all':

        sub_Vth = VsubVth[:, 100:]

    else:

        sub_Vth = VsubVth[sub_neurons, 100:]

    fig = plt.figure(figsize=(figsize_w,figsize_h))
    plt.pcolor(sub_Vth, cmap=cmap, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.xlim(0, nsteps - 100)
    plt.ylim(len(sub_Vth), 0)
    plt.xlabel('Time (10 ms)', fontsize = 15)
    plt.ylabel('Neuron Index Number', fontsize = 15)
    #plt.title('Voltage Dynamics: Motor Neurons', fontsize = 25)
    #plt.savefig('CElegansMotor')

def plot_dominant_modes(result_dict, sub_neurons):

    t = result_dict['t']
    nsteps = result_dict['steps']
    VsubVth = result_dict['v_solution']

    #os.chdir(path.data_4_analysis_dir)

    """ subset neurons """

    sub_Vth = VsubVth[100:, sub_neurons]

    """ Perform SVD """
    U, s, Z = np.linalg.svd(sub_Vth, full_matrices=True)

    normalized_s = np.divide(np.power(s, 2), np.sum(np.power(s, 2)))

    fig = plt.figure(figsize=(7,5))
    plt.scatter([1,2,3,4,5], normalized_s[:5], s = 120)
    plt.title('First Five Singular Values', fontsize = 20)
    plt.xlim(0, 6)
    plt.ylim(0, 1)
    plt.xlabel('Singular Value #', fontsize = 15)
    plt.show
    #plt.savefig('MotorNeurons_SingularVals')

    fig = plt.figure(figsize=(7,5))
    plt.plot(t[100:], U[:,0], t[100:], U[:,1], lw = 2)
    plt.title('First Two Dominant Modes Dynamics', fontsize = 20)
    plt.xlabel('Time (Seconds)', fontsize = 15)
    plt.legend(['1st mode', '2nd mode'], bbox_to_anchor=(0., .5, 1., .5), loc=1)
    plt.show
    #plt.savefig('MotorNeurons_PrincipalVecs')

    # Scaled scatterplots

    scaled_mode_1 = U[:,0] * s[0]
    scaled_mode_2 = U[:,1] * s[1]

    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(scaled_mode_1, scaled_mode_2, t[100:], lw = 3)
    plt.title('Phase Space of Two Modes (weighted)', fontsize = 20)
    plt.show

def plot_neurons_voltage(result_dict, neurons_list, index_Array, return_only_solution = False):

    t = result_dict['t']
    nsteps = result_dict['steps']
    VsubVth = result_dict['v_solution'].transpose()

    target_Neurons = VsubVth[index_Array, 100:]

    if return_only_solution == False:

        fig = plt.figure(figsize=(9,7))

        labels = []

        for neuron_index in index_Array:

            labels.append(neurons_list[neuron_index]['name'])

        k = 0

        while k < len(index_Array):

            plt.plot(t[100:], target_Neurons[k, :], lw = 2, label = labels[k])

            k += 1

        plt.title('Voltage Dynamics', fontsize = 15)
        plt.xlabel('Time (Seconds)')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1)
        plt.show
        plt.savefig('Voltage_Dynamics')

    else:

        return target_Neurons

def project_modes(p1, p2, result_dict, neurons_list, neurons_group):

    t = result_dict['t']
    nsteps = result_dict['steps']
    VsubVth = result_dict['v_solution'].transpose()

    target_index = []
    target_name = []

    for neuron in neurons_list:

        if neuron['group'] in neurons_group:

            target_index.append(neuron['index'])
            target_name.append(neuron['name'])

    target_neurons = VsubVth[target_index, 100:]

    projections_p1 = []
    projections_p2 = []

    for timeseries in target_neurons:

        proj_1 = np.sum(np.multiply(timeseries, p1))
        proj_2 = np.sum(np.multiply(timeseries, p2))

        projections_p1.append(proj_1)
        projections_p2.append(proj_2)

    return projections_p1, projections_p2, target_name

def plot_curvature(phi, phi_min, phi_max):

    fig = plt.figure(figsize=(10,7))

    plt.pcolor(phi, cmap = 'copper', vmin = phi_min, vmax = phi_max)
    plt.xlim(0, 23)
    plt.ylim(len(phi), 0)
    plt.colorbar()