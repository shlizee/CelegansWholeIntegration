---
layout: post
excerpt_separator: <!--more-->
title: Generating baseline locomotion - backward and forward
---


First locomotion movements that we generate with **_CelegansWholeIntegration_** are **forward and backward** movements. These movements are similar to translating sinusoidal. This video recorded in MIT experimental lab demonstrates the body postures involved (while there is variation in postures these are pretty typical).

<iframe width="560" height="315" src="https://www.youtube.com/embed/olrkWpCqVCE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

In this video, C. elegans initially moves forward and then when receives mechanical stimulus moves backward. Such response is called the tap response.  

validate our model for three basic locomotion patterns: forward movement, backward movement and turn. For each pattern we design a force wave travelling along the body with variable frequency to infer neural dynamics associated with it. These neural dynamics are then forward integrated by the nervous system to generate the body posture. When we simulate the integration, we observe that
 
generated movements can be almost indistinguishable from locomotion characteristics of freely moving animals (see SM Videos, snapshots, curvature maps, and calcium activity in Fig. 1). By measuring the error between the imposed traveling wave and the actual body curvature we observe that the nervous system includes preference for particular periods of the force, with optimal frequency being approximately 2 and 4s. These results indicate that the response of the nervous system is shaping the external force in a nontrivial and nonlinear manner. 

<!--more-->
