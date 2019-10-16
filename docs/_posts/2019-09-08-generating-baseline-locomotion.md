---
layout: post
excerpt_separator: <!--more-->
title: Generating baseline locomotion - forward and backward
---


First movements that we generate with **_CelegansWholeIntegration_** are **baseline forward and backward** locomotion movements. These movements are similar in shape to translating sinusoidals. The video below was recorded in an experimental lab and demonstrates the body postures involved:  

<iframe width="560" height="315" src="https://www.youtube.com/embed/olrkWpCqVCE" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

In this video, C. elegans initially moves forward and then when receives a mechanical stimulus (touch in the anterior part) switches direction and moves backward. While there is variation in postures these are pretty typical. Such a response is called the tap withdrawl.   

To generate the **baseline forward and backward** movements we design a **force wave travelling along the body** with variable frequency to infer neural dynamics associated with it. These neural dynamics are then forward integrated by the nervous system to generate the body postures. We are able to generate the following locomotion patterns (forward-left; backward-right): 

<iframe width="336" height="188" src="https://www.youtube.com/embed/UPrO7GtezbM" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>   <iframe width="336" height="188" src="https://www.youtube.com/embed/cilgztffR7w" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>  
  
When we simulate the integration, we observe that generated movements can be **very similar** to locomotion characteristics of freely moving animals (see above videos forward(left) and backward(right) ). By measuring the error between the imposed traveling wave and the actual body curvature we observe that the nervous system includes preference for particular periods of the force, with optimal frequency being approximately 2 and 4s. These results indicate that the **response of the nervous system is shaping the external force in a nontrivial and nonlinear manner**. 


