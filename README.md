# ACC2022 VMP-NARMAX

Experiments and supplementary derivations for the paper entitled

"Variational message passing for online polynomial NARMAX identification"

published at the [2022 American Control Conference](https://acc2022.a2c2.org/).

The goal is infer parameters in a polynomial NARMAX model (see [ForneyLab node code](https://github.com/biaslab/NARMAX)) and simulate outputs. Typically, (recursive) least-squares or another form of maximum-likelihood estimation is used. In this project, we employ variational [Free Energy Minimisation](https://en.wikipedia.org/wiki/Free_energy_principle) in the form of variational message passing on a Forney-style factor graph.

We run a series of verification experiments on data generated from a NARMAX system, comparing performance as a function of sample size, polynomial order and simulation noise.

#### Comments
Questions, comments and general feedback can be directed to the [issues tracker](https://github.com/biaslab/ACC2022-vmpNARMAX/issues).
