# ACC2022 VMP-NARMAX

Experiments and supplementary derivations for a submission to the [2022 American Control Conference](https://acc2022.a2c2.org/) entitled

"Variational message passing for online polynomial NARMAX identification".

The goal is infer parameters in a NARMAX model (see [ForneyLab node code](https://github.com/biaslab/NARMAX)) and simulate outputs. Typically, (recursive) least-squares or another form of maximum-likelihood estimation is used. In this project, we employ variational [Free Energy Minimisation](https://en.wikipedia.org/wiki/Free_energy_principle). To be precise, we pass variational messages on a Forney-style factor graph of the generative model. 

We run a series of verification experiments on data generated from a NARMAX system, comparing performance as a function of sample size, polynomial order and simulation noise.

#### Comments
Questions, comments and general feedback can be directed to the [issues tracker](https://github.com/biaslab/ACC2022-vmpNARMAX/issues).
