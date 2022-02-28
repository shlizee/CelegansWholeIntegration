# coding: utf-8

########################################################################################################################################################################
# VISCOELASTIC BODY MODEL SIMULATION MODULE ############################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

import os
import numpy as np
import scipy.io as sio
from imageio import imread
from scipy import integrate, interpolate, linalg
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dynworm import sys_paths as paths
from dynworm import body_params as b_params
from IPython.display import clear_output
import time

np.random.seed(10)

########################################################################################################################################################################
# BASE ENVIRONMENT INITIALIZATION ######################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
######################################################################################################################################################################## 

def initialize_params_body(custom_params = False):

    global params_obj_body

    if custom_params == False:

        params_obj_body = b_params.higher_fluid_density_default.copy()
        print('Using the default body parameters')

    else:

        assert type(custom_params) == dict, "Custom neural parameters should be of dictionary format"

        if validate_custom_body_params(custom_params) == True:

            params_obj_body = custom_params.copy()
            print('Accepted the custom body parameters')

def validate_custom_body_params(custom_params):

    # TODO: Also check for dimensions

    key_checker = []

    for key in b_params.default.keys():
        
        key_checker.append(key in custom_params)

    all_keys_present = np.sum(key_checker) == b_params.init_key_counts
    
    assert np.sum(key_checker) == b_params.init_key_counts, "Provided dictionary is incomplete"

    return all_keys_present


########################################################################################################################################################################
# ANIMATION FUNCTIONS ##################################################################################################################################################
# NEEDS REWORK #########################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

def produce_animation(x, y, filename, xmin, xmax, ymin, ymax, figsize_x, figsize_y, fps = 100, interval = 10, diameter_scaler = 1., \
    background_img_path = False, worm_color = 'black', facecolor = 'white', axis = 'off', \
    text_pos_dict = False, data_dict = False, worm_dict = b_params.worm_dict_default):

    """ Define the video template """

    fig = plt.figure(figsize=(figsize_x, figsize_y))
    fig.set_dpi(100)

    if background_img_path != False:
        img = imread(background_img_path)

    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))

    segments_count = len(x[0, :])

    """ Render the worm """

    patch_list = []

    diameters = diameter_scaler * b_params.h_interpolate(np.linspace(0, 24, segments_count))

    radius = np.divide(diameters, 1.5)

    """ Text placeholders """

    if text_pos_dict != False:

        ax_texts = {}

        for text_num in range(len(text_pos_dict)):

            text_x_coord = text_pos_dict[str(text_num)][0]
            text_y_coord = text_pos_dict[str(text_num)][1]
            text_fontsize = text_pos_dict[str(text_num)][2]
            text_color = text_pos_dict[str(text_num)][3]

            ax_texts["text_placeholder_{0}".format(text_num)] = ax.text(text_x_coord, text_y_coord, '', fontsize = text_fontsize, color = text_color)

    """ Re-painting worm """

    for k in range(0, segments_count):

        if type(worm_dict['worm_seg_color']) == str:

            patch_list.append(plt.Circle((x[0,k], y[0,k]), radius[k], color = worm_dict['worm_seg_color']))

        else:

            patch_list.append(plt.Circle((x[0,k], y[0,k]), radius[k], color = worm_dict['worm_seg_color'][k]))

    """ Initialize worm rendering """

    def init():

        for k in range(len(patch_list)):

            patch = patch_list[k]
            patch.center = x[0, k], y[0, k]
            ax.add_patch(patch)

        return patch_list

    """ Frame by frame animation function """

    def animate(i):

        """ Progress tracking """

        if i % 100 == 0:

            print("animated timepoint: " + str(i / 100) + " seconds")

        elif i == len(x)-1:

            print("redering complete!")

        """ Update worm segment positions """

        for k in range(0, segments_count):

            patch = patch_list[k]
            pos_x, pos_y = patch.center
            pos_x = x[i, k]
            pos_y = y[i, k]
            patch.center = (pos_x, pos_y)

        """ Update text + data info according to text_pos_dict and data_dict """

        if text_pos_dict != False and data_dict != False:

            for text_placeholder_num in range(len(ax_texts)):

                data_entry_exists = data_dict[str(text_placeholder_num)]['string'].find('data')

                if data_entry_exists != -1:

                    data_start_ind = data_entry_exists
                    data_end_ind = data_entry_exists + 4

                    text_part_1 = data_dict[str(text_placeholder_num)]['string'][:data_start_ind]
                    data_part =  str(data_dict[str(text_placeholder_num)]['data'][i])
                    text_part_2 = data_dict[str(text_placeholder_num)]['string'][data_end_ind:]

                    ax_texts["text_placeholder_" + str(text_placeholder_num)].set_text(text_part_1 + data_part + text_part_2)

                else:

                    ax_texts["text_placeholder_" + str(text_placeholder_num)].set_text(data_dict[str(text_placeholder_num)]['string'])

        """ Trail display """

        if worm_dict['display_trail'] == True:

            patch_list.append(plt.Circle((x[i,worm_dict['trail_point']], y[i,worm_dict['trail_point']]), worm_dict['trail_width'], \
                color = worm_dict['trail_color'], alpha = 0.7))

            patch = patch_list[-1]

            patch.center = x[i, worm_dict['trail_point']], y[i, worm_dict['trail_point']]

            ax.add_patch(patch)

        return patch_list

    """ Encode video """

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x), interval=interval, blit=True)
    ax.axis(axis)

    os.chdir(paths.vids_dir)

    """ Background image """

    if background_img_path != False:

        ax.imshow(img, zorder=0, extent=[xmin, xmax, ymin, ymax])

    """ Save video """

    anim.save(filename + '.mp4',savefig_kwargs={'facecolor':facecolor})

    os.chdir(paths.default_dir)

def produce_animation_swarm(x_list, y_list, filename, xmin, xmax, ymin, ymax, figsize_x, figsize_y, fps = 100, interval = 10, diameter_scaler = 1.2, \
    background_img_path = False, worm_color = 'black', facecolor = 'black', axis = 'on', \
    text_pos_dict = False, data_dict = False, worm_dict = b_params.worm_dict_default):

    """ Define the video template """

    fig = plt.figure(figsize=(figsize_x, figsize_y))
    fig.set_dpi(100)

    if background_img_path != False:
        img = imread(background_img_path)

    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))

    """ Render the worm """

    patch_list = []

    segments_count = len(x_list[0][0, :])

    diameters = diameter_scaler * b_params.h_interpolate(np.linspace(0, 24, segments_count))

    radius = np.divide(diameters, 1.5)

    """ Text placeholders """

    if text_pos_dict != False:

        ax_texts = {}

        for text_num in range(len(text_pos_dict)):

            text_x_coord = text_pos_dict[str(text_num)][0]
            text_y_coord = text_pos_dict[str(text_num)][1]
            text_fontsize = text_pos_dict[str(text_num)][2]
            text_color = text_pos_dict[str(text_num)][3]

            ax_texts["text_placeholder_{0}".format(text_num)] = ax.text(text_x_coord, text_y_coord, '', fontsize = text_fontsize, color = text_color)

    """ Re-painting worm """

    for swarm_count in range(len(x_list)):

        for k in range(0, segments_count):

            if type(worm_dict['worm_seg_color']) == str:

                patch_list.append(plt.Circle((x_list[swarm_count][0,k], y_list[swarm_count][0,k]), radius[k], color = worm_dict['worm_seg_color']))

            else:

                patch_list.append(plt.Circle((x_list[swarm_count][0,k], y_list[swarm_count][0,k]), radius[k], color = worm_dict['worm_seg_color'][k]))

    x_stacked = np.hstack(x_list)
    y_stacked = np.hstack(y_list)

    """ Initialize worm rendering """

    def init():

        for k in range(len(patch_list)):

            patch = patch_list[k]
            patch.center = x_stacked[0, k], y_stacked[0, k]
            ax.add_patch(patch)

        return patch_list

    """ Frame by frame animation function """

    def animate(i):

        """ Progress tracking """

        if i % 100 == 0:

            print("animated timepoint: " + str(i / 100) + " seconds")

        elif i == len(x_list[0])-1:

            print("redering complete!")

        """ Update worm segment positions """

        for k in range(len(patch_list)):

            patch = patch_list[k]
            pos_x, pos_y = patch.center
            pos_x = x_stacked[i, k]
            pos_y = y_stacked[i, k]
            patch.center = (pos_x, pos_y)

        """ Update text + data info according to text_pos_dict and data_dict """

        if text_pos_dict != False and data_dict != False:

            for text_placeholder_num in range(len(ax_texts)):

                data_entry_exists = data_dict[str(text_placeholder_num)]['string'].find('data')

                if data_entry_exists != -1:

                    data_start_ind = data_entry_exists
                    data_end_ind = data_entry_exists + 4

                    text_part_1 = data_dict[str(text_placeholder_num)]['string'][:data_start_ind]
                    data_part =  str(data_dict[str(text_placeholder_num)]['data'][i])
                    text_part_2 = data_dict[str(text_placeholder_num)]['string'][data_end_ind:]

                    ax_texts["text_placeholder_" + str(text_placeholder_num)].set_text(text_part_1 + data_part + text_part_2)

                else:

                    ax_texts["text_placeholder_" + str(text_placeholder_num)].set_text(data_dict[str(text_placeholder_num)]['string'])

        """ Trail display """

        if display_trail == True:

            for swarm_count in range(len(x_list)):

                patch_list.append(plt.Circle((x_list[swarm_count][i,-1], y_list[swarm_count][i,-1]), worm_dict['trail_width'], 
                    color = worm_dict['trail_color'], alpha = 0.7))

                patch = patch_list[-1]

                patch.center = x_list[swarm_count][i, worm_dict['trail_point']], y_list[swarm_count][i, worm_dict['trail_point']]

                ax.add_patch(patch)

        return patch_list

    """ Encode video """

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(x_list[0]), interval=interval, blit=True)
    ax.axis(axis)

    """ Background image """

    os.chdir(paths.vids_dir)

    if background_img_path != False:

        ax.imshow(img, zorder=0, extent=[xmin, xmax, ymin, ymax])

    """ Save video """

    anim.save(filename + '.mp4',savefig_kwargs={'facecolor':facecolor})

    os.chdir(paths.default_dir)

########################################################################################################################################################################
# MASTER FUNCTION FOR BODY MODEL SIMULATION ############################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

def solve_bodymodel(result_dict_network, xinit = 0, yinit = 0, orientation_angle = 0, custom_y_init = False, \
    kappa_scaling = 1, kappa_forcing = False, \
    use_simplified_calcium = True, custom_muscle_map = False, custom_force = False, return_steps_only = False, return_kappa_only = False, 
    xmin = -120, xmax = 80, ymin = -100, ymax = 100):

    # ACCEPT ANOTHER DICT OBJECT WITH BODY SIM CONFIGS

    assert 'params_obj_body' in globals(), "Body parameters must be initialized before running the simulation"

    if custom_muscle_map == False:

        params_obj_body['muscle_map'] = b_params.muscle_map_f.copy()

    else:
        #TODO: VALIDATE MUSCLEMAP

        params_obj_body['muscle_map'] = custom_muscle_map.copy()
        print("using custom muscle map")

    params_obj_body['kappa_scaling'] = kappa_scaling

    if type(kappa_forcing) != bool:

        params_obj_body['kappa_forcing'] = kappa_forcing

    else:

        params_obj_body['kappa_forcing'] = np.zeros(23)

    params_obj_body['t0'] = 0
    params_obj_body['tf'] = (len(result_dict_network['v_solution']) - 1) * params_obj_body['dt']

    t0 = params_obj_body['t0']
    tf = params_obj_body['tf']
    dt_network = result_dict_network['dt']
    dt_body = params_obj_body['dt']

    nsteps_network = int(np.floor((tf - t0)/dt_network) + 1)
    nsteps_body = int(np.floor((tf - t0)/dt_body) + 1) # Identical to len(v_solution)
    params_obj_body['nsteps_body'] = nsteps_body

    """ SET UP FORCE INTERPOLATE """
    timepoints_body = np.linspace(t0, tf, params_obj_body['nsteps_body'])
    params_obj_body['timepoints_body'] = timepoints_body

    steps = neuron_voltages_2_muscles(result_dict_network, custom_force, use_simplified_calcium)

    if return_steps_only == True:

        return steps

    compute_initcond(params_obj_body, xinit, yinit, orientation_angle, custom_y_init)
    compute_params(params_obj_body)

    if return_kappa_only == True:

        return compute_kappa()

    params_obj_body['interpolate_force'] = interpolate.interp1d(params_obj_body['timepoints_body'], params_obj_body['force'], axis=0, kind = 'nearest', fill_value = "extrapolate")

    InitCond = params_obj_body['IC']
    N = len(InitCond)

    progress_milestones = np.linspace(0, params_obj_body['nsteps_body'], 10).astype('int')

    """ Configuring the ODE Solver """
    r = integrate.ode(visco_elastic_rod_rhs).set_integrator('dopri5', rtol = 10e-5, atol = 10e-5, max_step = 0.001)
    r.set_initial_value(InitCond, t0)

    """ Additional Python step to store the trajectories """
    t = np.zeros(params_obj_body['nsteps_body'])
    Traj = np.zeros((params_obj_body['nsteps_body'], N))

    t[0] = t0
    Traj[0, :] = InitCond[:N]

    """ Integrate the ODE(s) across each delta_t timestep """

    print("Computing body movements...")

    k = 1

    while r.successful() and k < params_obj_body['nsteps_body']:

        r.integrate(r.t + dt_body)

        t[k] = r.t
        Traj[k, :] = r.y

        k += 1

        if k in progress_milestones:

            print(str(np.round((float(k) / params_obj_body['nsteps_body']) * 100, 1)) + '% ' + 'completed')

    x, y = solve_xy(Traj[:, 0], Traj[:, 1], Traj[:, 2:26])
    
    print("Post-processing the simulated body...")
    x_post, y_post = postprocess_xy(x, y)

    print("Rendering simulated body movements...")
    fig = plt.figure(figsize=(17,17))

    plt.plot(x_post[:, 0], y_post[:, 0],  linewidth = 5, color = 'black')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()

    result_dict_body = {
            "t" : t,
            "x" : x_post,
            "y" : y_post,
            "phi" : Traj[:, 2:26],
            "V2M_steps" : steps
            }

    return result_dict_body

########################################################################################################################################################################
# SIMULATION SPECIFIC ENVIRONMENT FUNCTIONS ############################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

def neuron_voltages_2_muscles(result_dict_network, custom_force = False, use_simplified_calcium = False):

    if type(custom_force) != bool:

        stacked_force = custom_force

        if use_simplified_calcium == True:

            force_dynamics = np.divide(np.power(15 * stacked_force, 2), 1 + np.power(15 * stacked_force, 2))
            force_dynamics = params_obj_body['scaled_factor'] * force_dynamics

        else:

            calcium_dynamics = muscle_activity_2_calcium(stacked_force)
            force_dynamics = np.divide(np.power(15 * calcium_dynamics, 2), 1 + np.power(15 * calcium_dynamics, 2))
            force_dynamics = params_obj_body['scaled_factor'] * force_dynamics

    else:

        VsubVth = result_dict_network['v_solution']

        muscle_input = np.zeros((len(VsubVth), len(params_obj_body['muscle_map'][:, 0])))

        for k in range(len(VsubVth)):

            muscle_input[k, :] = np.dot(params_obj_body['muscle_map'], VsubVth[k, :])

        dorsal_left = muscle_input[:, 0::4]
        dorsal_right = muscle_input[:, 1::4]
        ventral_left = muscle_input[:, 2::4]
        ventral_right = muscle_input[:, 3::4]

        dorsal_left_positive = np.multiply(dorsal_left, dorsal_left > 0)
        dorsal_right_positive = np.multiply(dorsal_right, dorsal_right > 0)
        ventral_left_positive = np.multiply(ventral_left, ventral_left > 0)
        ventral_right_positive = np.multiply(ventral_right, ventral_right > 0)

        dorsal_sum = np.add(dorsal_left_positive, dorsal_right_positive)
        ventral_sum = np.add(ventral_left_positive, ventral_right_positive)

        stacked_force = np.hstack([dorsal_sum, ventral_sum])

        if use_simplified_calcium == True:

            force_dynamics = np.divide(np.power(15 * stacked_force, 2), 1 + np.power(15 * stacked_force, 2))
            force_dynamics = params_obj_body['scaled_factor'] * force_dynamics

        else:

            calcium_dynamics = muscle_activity_2_calcium(stacked_force)
            force_dynamics = np.divide(np.power(15 * calcium_dynamics, 2), 1 + np.power(15 * calcium_dynamics, 2))
            force_dynamics = params_obj_body['scaled_factor'] * force_dynamics

        steps = {

            'voltage' : VsubVth,
            'muscle_input' : muscle_input,
            'dorsal_left' : dorsal_left,
            'dorsal_right' : dorsal_right,
            'ventral_left' : ventral_left,
            'ventral_right' : ventral_right,
            'stacked_force' : stacked_force,
            'force_dynamics' : force_dynamics

            }

        params_obj_body['force'] = force_dynamics

        return steps

def neuron_voltages_2_muscles_instant(v_solution_instant, ind):

    muscle_input = np.dot(params_obj_body['muscle_map'], v_solution_instant)

    dorsal_left = muscle_input[0::4]
    dorsal_right = muscle_input[1::4]
    ventral_left = muscle_input[2::4]
    ventral_right = muscle_input[3::4]

    dorsal_left_positive = np.multiply(dorsal_left, dorsal_left > 0)
    dorsal_right_positive = np.multiply(dorsal_right, dorsal_right > 0)
    ventral_left_positive = np.multiply(ventral_left, ventral_left > 0)
    ventral_right_positive = np.multiply(ventral_right, ventral_right > 0)

    dorsal_sum = np.add(dorsal_left_positive, dorsal_right_positive)
    ventral_sum = np.add(ventral_left_positive, ventral_right_positive)
    stacked_force = np.hstack([dorsal_sum, ventral_sum])

    scaled_muscles = np.divide(np.power(15 * stacked_force, 2), 1 + np.power(15 * stacked_force, 2))
    scaled_muscles = params_obj_body['scaled_factor'] * scaled_muscles

    params_obj_body['force'][ind, :] = scaled_muscles

def muscle_activity_2_calcium(stacked_force):

    stacked_force_interpolate = interpolate.interp1d(params_obj_body['timepoints_body'], stacked_force, axis=0, kind = 'nearest', fill_value = "extrapolate")
    params_obj_body['stacked_force_interpolate'] = stacked_force_interpolate
    dt_body = params_obj_body['dt']

    muscle_count = len(stacked_force[0, :])
    calcium_init = np.zeros(muscle_count * 4)

    r_calcium = integrate.ode(muscle_activity_2_calcium_rhs).set_integrator('vode', with_jacobian = True)
    r_calcium.set_initial_value(calcium_init, params_obj_body['t0'])

    calcium_traj = np.zeros((len(stacked_force), muscle_count * 4))
    calcium_traj[0, :] = calcium_init

    print("Computing calcium dynamics...")

    k = 1

    while r_calcium.successful() and k < params_obj_body['nsteps_body']:

        r_calcium.integrate(r_calcium.t + dt_body)

        calcium_traj[k, :] = r_calcium.y

        k += 1

    beta, beta_dot, eta, eta_dot = np.split(calcium_traj, 4, axis = 1)

    return eta

def compute_rotation_matrix(theta_degree):

    theta = np.radians(theta_degree)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))

    return theta, R

def compute_initcond(params_obj_body, xinit_custom = False, yinit_custom = False, orientation_angle = False, custom_y_init = False):

    # positive orientation angle --> counterclockwise rotation
    # negative orientation angle --> clockwise rotation

    xinit = np.linspace(0, 2 * np.pi, params_obj_body['segments_count'] + 1) + xinit_custom

    if type(custom_y_init) == bool:

        yinit = np.sin(xinit) + yinit_custom

    else:

        yinit = custom_y_init

    angle_radians, rotation_mat = compute_rotation_matrix(orientation_angle)

    xy_coords = np.vstack([xinit, yinit])
    xy_coords_transformed = np.dot(rotation_mat, xy_coords)

    x_init = xy_coords_transformed[0, :]
    y_init = xy_coords_transformed[1, :]

    phi_init = np.arcsin(np.diff(yinit)) + angle_radians

    initcond = np.zeros(52)
    initcond[0] = xinit[0]
    initcond[1] = yinit[0]
    initcond[2:26] = phi_init

    params_obj_body['IC'] = initcond

def compute_params(params_obj_body):

    h = np.ones(params_obj_body['segments_count'])
    h[20:] = 0.1
    #h = b_params.seg_lengths
    params_obj_body['h'] = h

    params_obj_body['b'] = params_obj_body['a']
    params_obj_body['area'] = np.pi * params_obj_body['a']**2

    params_obj_body['mp'] = params_obj_body['rho'] * params_obj_body['area'] * params_obj_body['h']
    params_obj_body['w'] = params_obj_body['a'] * params_obj_body['alpha']
    params_obj_body['I'] = np.pi * (params_obj_body['a']**4) / 4.
    params_obj_body['J'] = np.diag(params_obj_body['rho'] * params_obj_body['I'] * params_obj_body['h'])

    params_obj_body['EI'] = params_obj_body['E'] * params_obj_body['I']
    params_obj_body['v_bar'] = params_obj_body['E'] * np.pi / 8. * (params_obj_body['alpha']**3) #Stiffness 
    params_obj_body['v'] = params_obj_body['alpha'] * params_obj_body['a']**2 * params_obj_body['v_bar']

    diag_A = -1 * np.add(np.reciprocal(params_obj_body['mp'][:-1]), np.reciprocal(params_obj_body['mp'][1:]))
    lower_A = np.reciprocal(params_obj_body['mp'][1:-1])
    upper_A = np.reciprocal(params_obj_body['mp'][1:-1])

    A = np.diag(diag_A) + np.diag(lower_A, -1) + np.diag(upper_A, 1)
    A_plus = np.linalg.inv(A)
    A_plus_dimcount = len(A_plus[:,0])
    A_plus_ext = np.zeros((A_plus_dimcount + 1, A_plus_dimcount + 1))
    A_plus_ext[:A_plus_dimcount, :A_plus_dimcount] = A_plus
    params_obj_body['A_plus'] = A_plus_ext
    params_obj_body['kappa'] = np.zeros(params_obj_body['force'][:, :params_obj_body['segments_count'] - 1].shape)

def compute_params_spatial_grad(params_obj_body, nsteps_traj_body):

    h = np.ones(params_obj_body['segments_count'])
    h[20:] = 0.1
    params_obj_body['h'] = h

    params_obj_body['b'] = params_obj_body['a']
    params_obj_body['area'] = np.pi * params_obj_body['a']**2

    params_obj_body['mp'] = params_obj_body['rho'] * params_obj_body['area'] * params_obj_body['h']
    params_obj_body['w'] = params_obj_body['a'] * params_obj_body['alpha']
    params_obj_body['I'] = np.pi * (params_obj_body['a']**4) / 4.
    params_obj_body['J'] = np.diag(params_obj_body['rho'] * params_obj_body['I'] * params_obj_body['h'])

    params_obj_body['EI'] = params_obj_body['E'] * params_obj_body['I']
    params_obj_body['v_bar'] = params_obj_body['E'] * np.pi / 8. * (params_obj_body['alpha']**3)
    params_obj_body['v'] = params_obj_body['alpha'] * params_obj_body['a']**2 * params_obj_body['v_bar']

    diag_A = -1 * np.add(np.reciprocal(params_obj_body['mp'][:-1]), np.reciprocal(params_obj_body['mp'][1:]))
    lower_A = np.reciprocal(params_obj_body['mp'][1:-1])
    upper_A = np.reciprocal(params_obj_body['mp'][1:-1])

    A = np.diag(diag_A) + np.diag(lower_A, -1) + np.diag(upper_A, 1)
    A_plus = np.linalg.inv(A)
    A_plus_dimcount = len(A_plus[:,0])
    A_plus_ext = np.zeros((A_plus_dimcount + 1, A_plus_dimcount + 1))
    A_plus_ext[:A_plus_dimcount, :A_plus_dimcount] = A_plus
    params_obj_body['A_plus'] = A_plus_ext
    params_obj_body['kappa'] = np.zeros((nsteps_traj_body, params_obj_body['segments_count'] - 1))
    params_obj_body['force'] = np.zeros((nsteps_traj_body, params_obj_body['segments_count'] * 2))

def compute_kappa():

    """ interpoloate method """

    fR = params_obj_body['force'][:, :params_obj_body['segments_count']-1]
    fL = params_obj_body['force'][:, params_obj_body['segments_count']:-1]

    k = np.divide(4 * (fR - fL) * params_obj_body['w'][:-1], np.subtract(8 * params_obj_body['v'][:-1] * params_obj_body['w'][:-1]**2, np.multiply(fR + fL, np.power(params_obj_body['h'][:-1], 2))))

    return k

########################################################################################################################################################################
# POST-SIMULATION FUNCTIONS: SOLVE XY COORDINATES FOR ALL BODY SEGMENTS AND SMOOTHING MOVEMENTS ########################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

def solve_xy(x1, y1, phi):

    radii = np.divide(b_params.h, 2.)

    x_coords = np.zeros((len(phi), 24))
    y_coords = np.zeros((len(phi), 24))
    x_coords[:, 0] = x1
    y_coords[:, 0] = y1

    for k in range(1, len(b_params.h)):

        k_ = k - 1

        x_coords[:, k] = (b_params.h[k] / 2.) * (np.cos(phi[:, k_]) + np.cos(phi[:, k])) + x_coords[:, k_]
        y_coords[:, k] = (b_params.h[k] / 2.) * (np.sin(phi[:, k_]) + np.sin(phi[:, k])) + y_coords[:, k_]

    x = np.zeros(x_coords.shape)
    y = np.zeros(y_coords.shape)

    for k in range(len(phi)):

        x[k, :] = x_coords[k, :] - 0.5 * np.multiply(b_params.h, np.cos(phi[k, :]))
        y[k, :] = y_coords[k, :] - 0.5 * np.multiply(b_params.h, np.sin(phi[k, :]))

    return x, y

def solve_xy_individual(x1, y1, phi):

    radii = np.divide(b_params.h, 2.)

    x_coords = np.zeros(24)
    y_coords = np.zeros(24)
    x_coords[0] = x1
    y_coords[0] = y1

    for k in range(1, len(b_params.h)):

        k_ = k - 1

        x_coords[k] = (b_params.h[k] / 2.) * (np.cos(phi[k_]) + np.cos(phi[k])) + x_coords[k_]
        y_coords[k] = (b_params.h[k] / 2.) * (np.sin(phi[k_]) + np.sin(phi[k])) + y_coords[k_]

    x = np.subtract(x_coords, (0.5 * np.multiply(b_params.h, np.cos(phi))))
    y = np.subtract(y_coords, (0.5 * np.multiply(b_params.h, np.sin(phi))))

    return x, y

def smooth(y, box_pts):
    
    y_smooth = np.zeros(y.shape)
    box = np.ones(box_pts)/box_pts
    
    for k in range(len(y[0, :])):
        
        y_smooth[:, k] = np.convolve(y[:, k], box, mode='same')
    
    return y_smooth

def gaussian_smoothing(y, degree):

    y_smooth = np.zeros(y.shape)

    for k in range(len(y[0, :])):
        
        y_smooth[:, k] = gaussian_filter(y[:, k], degree)
    
    return y_smooth

def postprocess_xy(x, y):

    interpolate_x = interpolate.interp1d(np.arange(0, 24), x, axis=1, fill_value = "extrapolate")
    interpolate_y = interpolate.interp1d(np.arange(0, 24), y, axis=1, fill_value = "extrapolate")

    expanded_x = interpolate_x(np.linspace(0, 24, 192))
    expanded_y = interpolate_y(np.linspace(0, 24, 192))

    expanded_x_smoothed = gaussian_smoothing(expanded_x, 5)
    expanded_y_smoothed = gaussian_smoothing(expanded_y, 5)

    return expanded_x_smoothed, expanded_y_smoothed

########################################################################################################################################################################
# RIGHT-HAND SIDE FUNCTION FOR VISCOELASTIC ROD MODEL ##################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################
########################################################################################################################################################################

def visco_elastic_rod_rhs(t, y):

    """ Unpack y """

    xyphi, xyphi_dot = np.split(y, 2)

    x1 = xyphi[0]
    y1 = xyphi[1]
    phi = xyphi[2:]

    xdot1 = xyphi_dot[0]
    ydot1 = xyphi_dot[1]
    phidot = xyphi_dot[2:]

    """ Empty arrays for x, y, x_dot, ydot """

    xvec = np.zeros(params_obj_body['segments_count'])
    yvec = np.zeros(params_obj_body['segments_count'])
    xdot = np.zeros(params_obj_body['segments_count'])
    ydot = np.zeros(params_obj_body['segments_count'])

    """ Populate x, y, x_dot, y_dot """

    xvec[0] = x1
    yvec[0] = y1
    xdot[0] = xdot1
    ydot[0] = ydot1

    for j in np.arange(1, len(xvec)):

        j_ = j - 1

        xvec[j] = xvec[j_] + (params_obj_body['h'][j] / 2.) * (np.cos(phi[j_]) + np.cos(phi[j]))
        yvec[j] = yvec[j_] + (params_obj_body['h'][j] / 2.) * (np.sin(phi[j_]) + np.sin(phi[j]))
        xdot[j] = xdot[j_] - (params_obj_body['h'][j] / 2.) * (np.sin(phi[j_]) * phidot[j_] + np.sin(phi[j]) * phidot[j])
        ydot[j] = ydot[j_] + (params_obj_body['h'][j] / 2.) * (np.cos(phi[j_]) * phidot[j_] + np.cos(phi[j]) * phidot[j])

    """ Mapping t -> Left and right forces, or Ventral and dorsal forces """
    """ rounding to integer method"""

    t_int = np.max([1, int(np.round(t / params_obj_body['dt']))])

    """ interpoloate method """

    fR = params_obj_body['interpolate_force'](t)[:params_obj_body['segments_count']-1]
    fL = params_obj_body['interpolate_force'](t)[params_obj_body['segments_count']:-1]

    """ Computing velocity of local body movement """

    v_tan_norm_0 = np.multiply(np.cos(phi), xdot) + np.multiply(np.sin(phi), ydot)
    v_tan_norm_1 = np.multiply(-1 * np.sin(phi), xdot) + np.multiply(np.cos(phi), ydot)

    v_tan_norm = np.concatenate([v_tan_norm_0, v_tan_norm_1])
    v_tan = v_tan_norm[:params_obj_body['segments_count']]
    v_norm = v_tan_norm[params_obj_body['segments_count']:]

    """ Compute curvature k """

    k = np.divide(4 * (fR - fL) * params_obj_body['w'][:-1], np.subtract(8 * params_obj_body['v'][:-1] * params_obj_body['w'][:-1]**2, np.multiply(fR + fL, np.power(params_obj_body['h'][:-1], 2))))

    """ curvature scaling """
    k = np.multiply(k, params_obj_body['kappa_scaling'])
    k = k + params_obj_body['kappa_forcing']

    params_obj_body['kappa'][t_int, :] = k

    """ Phi_diff and phidot_diff """

    phi_diff = phi[1:] - phi[:-1]
    phi_dot_diff = phidot[1:] - phidot[:-1]

    """ Contact moment """

    M = np.zeros(params_obj_body['segments_count'])
    M[:-1] = params_obj_body['EI'][:-1] * (phi_diff - k) + (2 * params_obj_body['b'][:-1]**2 * params_obj_body['damping']) * np.divide(phi_dot_diff, params_obj_body['h'][:-1])
    Mdiff = np.append(M[0], np.diff(M))

    """ Second derivative computation """

    Hh_mat = np.diag(params_obj_body['h']) + np.diag(params_obj_body['h'][1:], 1)
    Gh_mat = np.diag(params_obj_body['h']) + np.diag(params_obj_body['h'][1:], -1)
    Gcos = np.multiply(Gh_mat / 2., Gmat(np.cos(phi)))
    Gsin = np.multiply(Gh_mat / 2., Gmat(np.sin(phi)))
    Hcos = np.multiply(Hh_mat / 2., Hmat(np.cos(phi)))
    Hsin = np.multiply(Hh_mat / 2., Hmat(np.sin(phi)))

    """ Normal force """

    F_N0 = np.multiply(params_obj_body['a'] * params_obj_body['rho_f'] * params_obj_body['C_N'] * np.abs(v_norm), v_norm)
    F_N1 = np.multiply(np.sqrt(8 * params_obj_body['rho_f'] * params_obj_body['a'] * params_obj_body['mu'] * np.abs(v_norm)), v_norm)
    F_N = np.add(F_N0, F_N1)

    """ Tan force """

    F_T = np.multiply(2.7 * np.sqrt(2 * params_obj_body['rho_f'] * params_obj_body['a'] * params_obj_body['mu'] * np.abs(v_norm)), v_tan)

    """ W computation """

    W0 = np.multiply(-1 * F_T, np.cos(phi)) + np.multiply(F_N, np.sin(phi))
    W1 = np.multiply(-1 * F_T, np.sin(phi)) - np.multiply(F_N, np.cos(phi))
    W = np.concatenate([W0, W1])
    Wx = W[:params_obj_body['segments_count']]
    Wy = W[params_obj_body['segments_count']:]

    """ W_diff computation """

    Wx_diff0 = np.multiply(np.divide(params_obj_body['h'][:-1], params_obj_body['mp'][:-1]), Wx[:-1])
    WX_diff1 = np.multiply(np.divide(params_obj_body['h'][1:], params_obj_body['mp'][1:]), Wx[1:])
    Wx_diff = np.subtract(Wx_diff0, WX_diff1)
    Wx_diff = np.append(Wx_diff, 0)

    Wy_diff0 = np.multiply(np.divide(params_obj_body['h'][:-1], params_obj_body['mp'][:-1]), Wy[:-1])
    Wy_diff1 = np.multiply(np.divide(params_obj_body['h'][1:], params_obj_body['mp'][1:]), Wy[1:])
    Wy_diff = np.subtract(Wy_diff0, Wy_diff1)
    Wy_diff = np.append(Wy_diff, 0)

    """ Second derivative of Phi """

    phi_ddot_numerator0 = np.dot(np.dot(np.dot(-Gcos, params_obj_body['A_plus']), Hsin) + np.dot(np.dot(Gsin, params_obj_body['A_plus']), Hcos), np.power(phidot, 2))
    phi_ddot_numerator1 = Mdiff + np.dot(np.dot(Gcos, params_obj_body['A_plus']), Wy_diff) - np.dot(np.dot(Gsin, params_obj_body['A_plus']), Wx_diff)
    phi_ddot_numerator = np.add(phi_ddot_numerator0, phi_ddot_numerator1)

    phi_ddot_denominator = params_obj_body['J'] - np.dot(np.dot(Gcos, params_obj_body['A_plus']), Hcos) - np.dot(np.dot(Gsin, params_obj_body['A_plus']), Hsin)
    phi_ddot = np.linalg.solve(phi_ddot_denominator, phi_ddot_numerator)

    """ Second derivative of x and y """

    C_ddot = np.zeros(params_obj_body['segments_count'])
    cs1 = np.cos(phi[0]) * phidot[0]**2 + np.sin(phi[0]) * phi_ddot[0]
    cs2 = np.cos(phi[1]) * phidot[1]**2 + np.sin(phi[1]) * phi_ddot[1]
    C_ddot[1] = -params_obj_body['h'][1] / 2. * (cs1 + cs2)

    S_ddot = np.zeros(params_obj_body['segments_count'])
    sc1 = np.sin(phi[0]) * phidot[0]**2 - np.cos(phi[0]) * phi_ddot[0]
    sc2 = np.sin(phi[1]) * phidot[1]**2 - np.cos(phi[1]) * phi_ddot[1]
    S_ddot[1] = -params_obj_body['h'][1] / 2. * (sc1 + sc2)

    for j in range(2, params_obj_body['segments_count']):

        cs_i = np.cos(phi[j]) * phidot[j]**2 + np.sin(phi[j]) * phi_ddot[j]
        cs_im1 = np.cos(phi[j-1]) * phidot[j-1]**2 + np.sin(phi[j-1]) * phi_ddot[j-1]

        sc_i = np.sin(phi[j]) * phidot[j]**2 - np.cos(phi[j]) * phi_ddot[j]
        sc_im1 = np.sin(phi[j-1]) * phidot[j-1]**2 - np.cos(phi[j-1]) * phi_ddot[j-1]

        C_ddot[j] = C_ddot[j-1] - params_obj_body['h'][j] / 2. * cs_im1 - params_obj_body['h'][j] / 2. * cs_i
        S_ddot[j] = S_ddot[j-1] - params_obj_body['h'][j] / 2. * sc_im1 - params_obj_body['h'][j] / 2. * sc_i

    m_sum = np.sum(params_obj_body['mp'])

    x_ddot_1 = np.reciprocal(m_sum) * np.sum(np.multiply(params_obj_body['h'], Wx) - np.multiply(params_obj_body['mp'], C_ddot))
    y_ddot_1 = np.reciprocal(m_sum) * np.sum(np.multiply(params_obj_body['h'], Wy) - np.multiply(params_obj_body['mp'], S_ddot))

    output_dot = np.asarray([xdot[0], ydot[0]])
    output_dot = np.concatenate([output_dot, phidot])
    output_ddot = np.asarray([x_ddot_1, y_ddot_1])
    output_ddot = np.concatenate([output_ddot, phi_ddot])

    output_final = np.concatenate([output_dot, output_ddot])

    return output_final

def muscle_activity_2_calcium_rhs(t, y):

    # x1 = beta, x2 = beta_dot, x3 = eta, x4 = eta_dot

    x1, x2, x3, x4 = np.split(y, 4)

    u_t = params_obj_body['stacked_force_interpolate'](t)

    x1_dot = x2
    x2_dot = -(params_obj_body['c1'] * x2) - (params_obj_body['c2'] * x1) + (params_obj_body['c3'] * u_t)

    x3_dot = x4
    x4_dot = -(params_obj_body['c4'] * x4) - (params_obj_body['c5'] * x3) + (params_obj_body['c6'] * x1)

    return np.concatenate((x1_dot, x2_dot, x3_dot, x4_dot))

def Gmat(p):

    G = np.diag(p) + np.diag(p[1:], -1)

    return G

def Hmat(p):

    H = np.diag(np.append(p[:-1], 0)) + np.diag(p[1:], 1)

    return H