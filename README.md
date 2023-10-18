# Masters_thesis

This work was done during my Master's project at the University of Geneva under the supervision of Federico Sanchez. 
The objective was to find alternative ways of achieving Bayesian inference using ML tools.  
It uses the T2K near detector framework as a working example. We task the model to learn the mapping between a set of observables (outgoing muon momentums and angles) and the posterior distribution of latent variables (energy bin values of the neutrino flux at the near detector).
This is done using two different architectures, one using exclusively CNN (see the folder "Reweight_learning_with_CNN") and another one using normalizing flows conditionned by the set of observables (see the folder "Posterior_density_estimation_with_normalizing flows").
The second model uses the implementation for normalizing Flows in Pytorch of Durkan et al. "nflows" (https://github.com/bayesiains/nflows).
