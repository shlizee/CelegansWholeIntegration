---
layout: post
excerpt_separator: <!--more-->
title: Investigation of EI Imbalance on C. elegans Locomotion
---

A balance between excitatory and inhibitory signaling (EI balance) in the brain is essential for maintaining normal brain function in animals including C. elegans. Proper EI balance allows for the C. elegans to have normal locomotion. C. elegans mutants, namely zf-35, express excessive excitatory signals, causing the worm to exhibit seizures and spontaneous backward pauses and reversals during locomotion [1]. To further investigate the link between EI imbalance and its associated behavioral effects on C. elegans, we use the C. elegans Whole Integration model to reproduce the zf-35 mutant using 3 methods: increased gains to backwards, backwards and forwards, and all excitatory neurons.

![zf-35 simulations](/CelegansWholeIntegration/media/EI_Imbalance.png)

Emulating zf-35 mutant by increasing gains into AVA, AVD, AVE (backward neurons), AVA, AVD, AVE, PVC, RIM, AVB (backward + forward neurons) and all excitatory neurons. A: Instantaneous velocities (wormlength/s) of worm in 90 seconds; B: Mean velocity and number of reversals in 90 seconds.

<!--more-->

<iframe src="https://drive.google.com/file/d/1qNcRgfUkXMQYn_hWGBxUG8iZgdRJcJK4/preview" width="720" height="405" allow="autoplay"></iframe>

The three methods we considered are implemented as follows: for increased gains to backward neurons, we use a gain of 1275 pA into AVA, AVD, AVE interneurons; for increased gains to both backward and forward neurons, we use a gain of 600 pA into AVA, AVD, AVE, PVC, RIM, AVB interneurons; for increased gains to all excitatory neurons, we use a gain of 6 pA into all glutamergic and cholinergic neurons. The gain amplitudes are chosen to induce spontaneous reversals and pauses during locomotion behavior similar to those in-vivo zf-35.

From Figure 1, we noticed all three methods have impacts on locomotion but increasing gains to all excitatory neurons significantly decreased the mean velocity while increasing the frequency of reversals. In particular, the frequency of reversals increased by a factor of x6 which is similar to the factor measured during in-vivo experiments [1]. It is also worth noting that increasing gains to backward and forward neurons aren’t sufficient to produce similar behavioral effects - while they produced spontaneous reversals during the first few seconds, the worm quickly restored the velocity and rate of reversals similar to those of wild-type variant.

We also found that each scenario had a gain threshold where any higher gain would disable the worm’s movement entirely. The threshold for each of the three methods were found to be 1300 pA, 700 pA, and 100 pA respectively. The observed differences in thresholds are likely associated with the number of neurons receiving stimulus gains in each scenario: 6 for backward neurons only, 12 for both backward and forward neurons, and 253 for all excitatory neurons. With more neurons being stimulated, less stimulus gain per neuron was needed to saturate the nervous system and disable locomotion. 

In summary, our studies showcase a systematic method for C. elegans model to reproduce EI imbalance exhibited by zf-35 mutant. To emulate such a mutant, we increase stimulus gains to backward, forward, and all excitatory neurons. In particular, increasing gains to all excitatory neurons of glutamergic and cholinergic type causes the worm to exhibit abnormal locomotion behavior with increased frequency of reversals similar to in-vivo experiments. As our investigation was limited to broad neuron groups, however, further extensions of the method aiming to delineate the exact neurons/synapses with excitatory gains will allow us to reproduce behaviors associated with EI imbalance closer to in-vivo, 

**References**

[1] Yung-Chi Huang, Jennifer K Pirri, Diego Rayes, Shangbang Gao, Ben Mulcahy, Jeff Grant, Yasunori Saheki, Michael M Francis, Mei Zhen, Mark J Alkema (2019) Gain-of-function mutations in the UNC-2/CaV2α channel lead to excitation-dominant synaptic transmission in Caenorhabditis elegans eLife 8:e45905

