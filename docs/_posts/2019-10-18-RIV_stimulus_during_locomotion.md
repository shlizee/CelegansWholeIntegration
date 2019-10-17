---
layout: post
excerpt_separator: <!--more-->
title: A worm in a box? Triggering sharp turns with RIV pulses
---

#### Sharp Ventral Turns: forward locomotion is interrupted by RIV stimulus

RIV motorneurons synapse onto ventral neck muscles and have been associated with the execution of sharp ventral head bend. A **COOL** demo of this is an in-vivo scenario by [Alkema lab](https://www.umassmed.edu/AlkemaLab/) showing that they can confine _C. elegans_ in a rectangular region by stimulating RIV each time the worm reaches the boundary of that region. 

With _CelegansWholeIntegration_ we wanted to explore the effects of additional neural stimuli on forward locomotion. Such a scenario is a great test case to see if we can reproduce it. We set the _CelegansWholeIntegration_ worm to perfrom forward locomotion, i.e., without pulses of RIV stimuli the trajectory of the worm would move with in the same direction as it was initialized. Then while the worm moves forward we **stimulate RIV neurons (for 3 seconds) each time the anterior part is about to reach the boundary of a white region** and we generate the video below:

<iframe width="450" height="400" src="https://www.youtube.com/embed/hJfpxhMVUgc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<!--more-->
As can be seen in the video, each RIV stimulus causes sharp ventral bend of the head leading to a rotation of forward locomotion course by approximately 90 degrees while sustaining locomotion in the forward direction. This way the worm is trapped in the white rectangular region! When we inspect voltage activity of motorneurons, we observe that the turn corresponds to a bias added to the voltage activity of oscillating motor neurons.

![RIV Response](/CelegansWholeIntegration/media/RIV.png)

Inspection of how rotation is exhibited indicates that there are two posture states: **(i) neck straightening followed by a (ii) ventral turn**. These states are observed in experimental studies of the escape response as well. We investigate these states by performing dimension reduction on neural activity in each state and identify dominant neurons associated with the activity. We then compute the force magnitude resulting from dominant motor neurons activity. We find that during neck straightening state, dorsal and ventral forces are well balanced and cancel each other out, while in the ventral turn state there is a strong ventral force acting on the muscle segments. Such analysis reveals neural participation on the cellular level in each state and how neural activity is superimposed to create particular posture.

![RIV States](/CelegansWholeIntegration/media/RIVstates.png)

**_CelegansWholeIntegration_** allows to perfrom various ablations to the somatic connectome and compare **wild(non-ablated)** with **ablated** connectome for the same response. We therefore utilize ablations to seek which neurons would be most correlated (or actually anti-correlated) with sharp turns. In other words we are seeking for **neurons which ablation would significantly alter the RIV response**. We select all pairs of neurons (R and L) from the group of neurons **directly connected to RIV and separately ablate each pair**. The analysis shows that ablation of SMDV causes the most prominent change in the dynamics: the ablation disables the turn behavior and causes the body to continue with forward movement. As demostrated in the video below. 

<iframe width="450" height="400" src="https://www.youtube.com/embed/iuIAXnbTQKk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

Neural activity in the case of SMDV ablation is similar to neural activity during forward movement. 

![SMDVAblated RIVResponse](/CelegansWholeIntegration/media/RIV_SMDVAbl.png)

Indeed we see that SMDV ablation causes the worm to ingore RIV pulses. **It effectively caused the worm go get out of the box!** Overall these results suggest that both SMDV and RIV neurons are required to facilitate a sharp turn.

