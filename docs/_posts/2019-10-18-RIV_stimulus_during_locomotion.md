---
layout: post
excerpt_separator: <!--more-->
title: Worm in a box? Triggering sharp turns with RIV pulses
---

#### Sharp Ventral Turns: forward locomotion is interrupted by RIV stimulus

RIV motorneurons synapse onto ventral neck muscles and have been associated with the execution of sharp ventral head bend. A **COOL** demo of this is an in-vivo scenario by [Alkema lab](https://www.umassmed.edu/AlkemaLab/) showing that they can confine a worm in a rectangular region by stimulating RIV every time the worm reaches the boundary of that region. With _CelegansWholeIntegration_ we wanted to explore the effects of additional neural stimuli on forward locomotion so we chose this scenario as a test case and generate the video below. In this scenario we stimulate RIV neurons (for 3 seconds) every time it's about to reach the boundary of a white region. Without the RIV stimuli pulses the worm would perform forward locomotion and move straight. See what happens in the video below:

<iframe width="560" height="315" src="https://www.youtube.com/embed/hJfpxhMVUgc" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

<!--more-->
As can be seen in the video, each RIV stimulus causes sharp ventral bend of the head leading to a rotation of forward locomotion course by approximately 90 degrees while sustaining locomotion in the forward direction. 

![RIV Response](/CelegansWholeIntegration/media/RIV.png)

Neural activity indicates that the turn corresponds to a bias added to the voltage activity of oscillating motor neurons.
