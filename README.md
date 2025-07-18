## Neuromatch Academy project for the Deep Learning course

**EVERYTHING IS UNFINISHED, NOTHING WORKS!**

Can we create a model that successfully reduces the dimensionality of V1 neural activity (encodes) and uses the resulting features to predict grating visual stimuli (decodes)?

Using the Allen Visual Coding dataset: https://observatory.brain-map.org/visualcoding/ (or https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html)

Original Dataset Paper:
https://www.nature.com/articles/s41586-020-03171-x


*Data*: stimulus x neuron x time bin format, number of spikes in bin is the value. We can also get different stimuli properties and IDs to predict. 
*Model*: Undecided so far, some type of autoencoder with classifier on top? MLPs, CNNs, RNNs, AEs, VAEs, ...
- Dimension reduction, signal reconstruction, category prediction

Example hypothesis:
*The optimal number of latent variables to predict stimulus category is less than the number of input neurons*
- H0: n(latent) < n(neurons)