# CDC2021 VMP-NARMAX

Experiments and derivations for a submission to the [2021 Conference on Decision and Control](https://2021.ieeecdc.org/) entitled

"Online Bayesian NARMAX identification: a free energy minimisation approach".

The goal is infer parameters in a NARMAX model (see [ForneyLab node code](https://github.com/biaslab/NARMAX)). Typically, (recursive) least-squares or another form of frequentist estimation is used. In this project, we employ variational [Free Energy Minimisation](https://en.wikipedia.org/wiki/Free_energy_principle). To be precise, we employ variational message passing on a Forney-style factor graph of the generative model. We run a series of verification experiments and a validation experiment on the Silverbox data set from the [Nonlinear Benchmark](https://sites.google.com/view/nonlinear-benchmark/).

#### Comments
Questions, comments and general feedback can be directed to the [issues tracker](https://github.com/biaslab/CDC2021-vmpNARMAX/issues).
