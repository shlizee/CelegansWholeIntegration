---
layout: post
excerpt_separator: <!--more-->
title: A worm in a box? Triggering sharp turns with RIV pulses
---

#### Sharp Ventral Turns: forward locomotion is interrupted by RIV stimulus

RIV motorneurons synapse onto ventral neck muscles and have been associated with the execution of sharp ventral head bend. A **COOL** demo of this is an in-vivo scenario by [Alkema lab](https://www.umassmed.edu/AlkemaLab/) showing that they can confine _C. elegans_ in a rectangular region by stimulating RIV each time the worm reaches the boundary of that region. 

With _CelegansWholeIntegration_ we wanted to explore the effects of additional neural stimuli on forward locomotion. Such a scenario is a great test case to see if we can reproduce it. We set the _CelegansWholeIntegration_ worm to perfrom forward locomotion, i.e., without pulses of RIV stimuli the trajectory of the worm would move with in the same direction as it was initialized. Then while the worm moves forward we **stimulate RIV neurons (for 3 seconds) each time the anterior part is about to reach the boundary of a white region** and we generate the video below:

<iframe width="560" height="315" src="https://www.youtube.com/embed/hJfpxhMVUgc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<!--more-->
As can be seen in the video, each RIV stimulus causes sharp ventral bend of the head leading to a rotation of forward locomotion course by approximately 90 degrees while sustaining locomotion in the forward direction. This way the worm is trapped in the white rectangular region!

![RIV Response](/CelegansWholeIntegration/media/RIV.png)

When we inspect voltage activity of motorneurons, we observe that the turn corresponds to a bias added to the voltage activity of oscillating motor neurons.
