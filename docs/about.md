---
layout: page
title: About C. elegans Whole Integration
---

# _CelegansWholeIntegration_ project

_CelegansWholeIntegration_ project introduces a novel and one of a kind model to computationally emulate the whole somatic nervous system and its response to stimuli, and connect it with body dynamics for the nematode Caenorhabditis elegans (C. elegans). 
The model incorporates the full anatomical wiring diagram, somatic connectome, intracellular and extracellular neural dynamics. The model includes layers which translate neural activity to muscle forces and muscle impulses to body postures. 
In addition, it implements inverse integration which modulates neural dynamics according to external forces on the body. 
We test the model by comparing outcomes of body dynamics with in vivo experiments of the touch response and implement ablations found effective in those experiments. 
Furthermore, we perform additional ablations to elucidate novel details on these experiments.

# _CelegansWholeIntegration_ blog

Since the project facilitates exploration of various stimualtions and behavioral scenarios we have started a blog where we will describe the scenarios that we have studied. Our plan is to explore additional scenarios that will enhance the model and potentially contribute to better understanding of mappping neural dynamics to body movements. Through the blog we also aim to interact with the community and get it involved in identifying novel scenarios.

# _CelegansWholeIntegration_ code
We are working on the release of our code that will be entirely **open source**. The code repository will include the connectomics and simulation of neural dynamics, jointly called the nervous system module. In addition to the nervous system module we are releasing the body dynamics module. Both modules interface with each other and allow to simulate them both and close the loop between neural dynamics and body. We will also provide an API for using various components in the modules and developing your own simulations.

# _CelegansWholeIntegration_ manuscript
Please cite the _CelegansWholeIntegration_ preprint if you use this code:

```
@article{kim2019whole,  
  title={Whole integration of neural connectomics, dynamics and bio-mechanics for  
         identification of behavioral sensorimotor pathways in Caenorhabditis elegans},  
  author={Kim, Jimin and Santos, Julia A and Alkema, Mark J and Shlizerman, Eli},  
  journal={bioRxiv},  
  pages={724328},  
  year={2019},  
  publisher={Cold Spring Harbor Laboratory}  
}
```



