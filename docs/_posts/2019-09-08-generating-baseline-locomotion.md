---
layout: post
excerpt_separator: <!--more-->
title: Generating baseline locomotion - backward and forward
---


First movements that we generate with **_CelegansWholeIntegration_** are **baseline forward and backward** locomotion movements. These movements are similar in shape to translating sinusoidals. This video recorded in an experimental lab demonstrates the body postures involved (while there is variation in postures these are pretty typical):  

<iframe width="560" height="315" src="https://www.youtube.com/embed/olrkWpCqVCE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

In this video, C. elegans initially moves forward and then when receives a mechanical stimulus (tap in posterior part) switches direction and moves backward. Such a response is called the tap withdrawl.  

We use the generated **baseline forward and backward** movements to validate our model. For each pattern we design a force wave travelling along the body with variable frequency to infer neural dynamics associated with it. These neural dynamics are then forward integrated by the nervous system to generate the body posture. We are able to generate the following locomotion patterns: 

<iframe width="336" height="188" src="https://www.youtube.com/embed/UPrO7GtezbM" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>   <iframe width="336" height="188" src="https://www.youtube.com/embed/cilgztffR7w" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>  


| Forward | Backward |
|:-----------------------:|:------------------------:|


When we simulate the integration, we observe that generated movements can be almost indistinguishable from locomotion characteristics of freely moving animals (see above videos forward(left) and backward(right) ). By measuring the error between the imposed traveling wave and the actual body curvature we observe that the nervous system includes preference for particular periods of the force, with optimal frequency being approximately 2 and 4s. These results indicate that the response of the nervous system is shaping the external force in a nontrivial and nonlinear manner. 

<!--more-->
