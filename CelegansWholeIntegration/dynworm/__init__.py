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
		# a. Forces to dynamic body postures using fluid equation *** (Needs to be re-written in Julia)                     # 
#############################################################################################################################

# Computation layers  #######################################################################################################
# 1. Gap Junction Adjacency Matrix                            (279 * 279)                                                   #
# 2. Synaptic Adjacency Matrix                                (279 * 279)                                                   #
# 3. Excitatory/Inhibitory Map (Glutamate, Choline, GABA)     (279 * 279)                                                   #
# 4. Ion Channels                                                                  (AWA and AVL neurons)                    #
# 5. Extra-synaptic                                           (279 * 279)          (Planned)                                #
#############################################################################################################################

__author__ = 'Jimin Kim: jk55@u.washington.edu'
__version__ = '0.1.0'

from dynworm import sys_paths 
from dynworm import neural_params
from dynworm import body_params 
from dynworm import neurons_idx 
from dynworm import network_sim 
from dynworm import body_sim
#from dynworm import spatialgrad_sim
from dynworm import plotting