# coding: utf-8

# Model Components ##########################################################################################################
##### 1. Nervous System Simulation																		                    # 
		# a. Nembrane voltage simulation to Dynamic Stimuli (Leaky membrane + Gap + Synaptic + External stimulus)	        #
		# b. Resting potential (Vth) computation via network-wise fixed point computation									#
		# c. Simulation with External Forcing															                    #
		# d. Simulation with Proprioceptive Feedback 													                    #
##### 2. Muscle Force Simulation																	                        #
		# a. Neurons' voltages to Muscle Activity													                        #
		# b. Muscle Activity to Calcium dynamics														                    #
		# c. Calcium dynamics to Force                                									                    #
##### 3. Body Posture Simulation																		                    #
		# a. Forces to dynamic body postures using fluid equation                     # 
#############################################################################################################################

# Computation layers  #######################################################################################################
# 1. Gap Junction Adjacency Matrix                            (279 * 279)                                                   #
# 2. Synaptic Adjacency Matrix                                (279 * 279)                                                   #
# 3. Excitatory/Inhibitory Map (Glutamate, Choline, GABA)     (279 * 279)                                                   #
# 4. Ion Channels                                             (AWA and AVL neurons)(Uses Julia to solve)                    #
# 5. Extra-synaptic                                           (279 * 279)          (Planned)                                #
#############################################################################################################################

# TODO: USE JULIA INTEGRATOR FOR NON-LINEAR SIMULATIONS + BODY

__author__ = 'Jimin Kim: jk55@u.washington.edu'
__version__ = '0.2.0'

from dynworm import sys_paths # Handles working directory for different OS
from dynworm import neural_params # Contains connectomes + physiological parameters used for neural simulations
from dynworm import body_params # Contains body/surrounding liquid parameters used for body simulations
from dynworm import neurons_idx # Set of pre-defined indices for different neuron groups 
from dynworm import network_sim # Handles neural simulations
from dynworm import body_sim # Handles body simulations
#from dynworm import spatialgrad_sim # Real time simulations of environmental interactions. This code is confidential for now.
from dynworm import plotting # Suite of plotting functions
from dynworm import utils # Suite of various utility functions