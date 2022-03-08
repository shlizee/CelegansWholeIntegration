---
layout: post
excerpt_separator: <!--more-->
title: Changing locomotion direction through additional pulses of neural stimuli (Avoidance)
---

Once we have generated **baseline forward and backward** locomotion movement with **_CelegansWholeIntegration_** we explore the effects of stimulating additional neurons (with short pulses) during locomotion. First scenarios that we explored are neural triggers identified experimentally to change the course of movement. In this post we describe the avoidance pulse: stimulation of ALM + AVM.

#### Avoidance: forward locomotion is interrupted by ALM+AVM stimulus

When ALM+AVM are stimulated in-vivo during forward locomotion, _C. elegans_ reacts to the change by stopping and reversing. In the video below we **initiate forward locomotion** and **after 6 seconds apply ALM+AVM neural stimulation for 2 seconds**. 

<iframe width="560" height="315" src="https://www.youtube.com/embed/klOJb0DDGGU" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
<!--more-->
As can be seen in the video, ALM+AVM neural stimulus is capable to generate the avoidance behavior; the stimulus disturbs forward locomotion and initiates backward locomotion. Inspection of neural activity of motor neurons (DB neurons are Anterior->Posterior ordered; see image below) indicates that the stimulus induces a change in the directionality of neural activity traveling wave from Anterior->Posterior to Posterior->Anterior. The transition is marked through high constant activity in the anterior motor-neurons.

![ALMAVM Response](/CelegansWholeIntegration/media/ALMAVM.png)
