# Model Components ##########################################################################################################
##### 1. Nervous System Simulation																		                    #
		# a. Nembrane voltage simulation to Dynamic Stimuli (Leak + Ion + Gap + Synaptic + External stimulus)	            #
		# b. Resting potential (Vth) computation via network-wise fixed point computation									#
		# c. Simulation with External Forcing															                    #
		# d. Simulation with Proprioceptive Feedback 													                    #
##### 2. Muscle Force Simulation																	                        #
		# a. Neurons' voltages to Muscle Activity													                        #
		# b. Muscle Activity to Calcium dynamics														                    #
		# c. Calcium dynamics to Force                                									                    #
##### 3. Body Posture Simulation																		                    #
		# a. Forces to dynamic body postures using fluid equation                                                           #
##### 4. Standalone Neuron Simulation series                                                                                #
		# a. Ca,T and Ca,P model                                                                                            #
		# b. Generic neuron model                                                                                           #
#############################################################################################################################

# Computation layers  #######################################################################################################
# 1. Gap Junction Adjacency Matrix                            (279 * 279)                                                   #
# 2. Synaptic Adjacency Matrix                                (279 * 279)                                                   #
# 3. Excitatory/Inhibitory Map (Glutamate, Choline, GABA)     (279 * 279)                                                   #
# 4. Ion Channels                                             (AWA and AVL neurons)(Uses Julia to solve)                    #
# 5. Extra-synaptic                                           (279 * 279)          (Planned)                                #
#############################################################################################################################

# UPGRADE PLANS #############################################################################################################

# PHASE 1: Functional Spatialgradient code
## Parametrize visco_elastic_rod_rhs_julia -- Done
## Test Body simulation -- Done
## Rewrite solve_body_spatialgrads -- Done
## Test spatial gradient simulation -- Done
## Construct spatial gradient tutorial notebook -- Done

# PHASE 2: Parametrize all Julia solvers
## Parametrize AWA and AVL simulations with numba
## Parametrize all rhs with Julia solvers with numba
## Simpler implementation for adding non-linear channels to AWA and AVL
## Test AWA, AVL neuron simulations
## Add support for AWA and AVL for spatialgradient simulation

# PHASE 3 Modular design for Network simulation
## Construct computation graph for the full model
## Construct notebook for network sim and body sim with intended design
## Construct code structure that adheres to above with the goal of simple, robust and scalable
## Outline tasks to be taken for refactoring network_sim, body_sim and spatialgradient_sim

# PHASE 4 Modular design for Body/Spatial Gradient simulation
## TBA

#############################################################################################################################

__author__ = 'Jimin Kim: jk55@u.washington.edu, Linh Truong: linhtruong2001@gmail.com'
__version__ = '0.3.5'

from dynworm import sys_paths # Handles working directory for different OS
from dynworm import neural_params # Contains connectomes + physiological parameters used for neural simulations
from dynworm import body_params # Contains body/surrounding liquid parameters used for body simulations
from dynworm import neurons_idx # Set of pre-defined indices for different neuron groups 
from dynworm import network_sim # Handles neural simulations
from dynworm import body_sim # Handles body simulations
from dynworm import spatialgrad_sim # Real time simulations of environmental interactions. This code is confidential for now.
from dynworm import plotting # Suite of plotting functions
from dynworm import utils # Suite of various utility functions