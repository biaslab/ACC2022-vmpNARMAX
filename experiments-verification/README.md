## Verification experiments

The experiments are implemented in a combination of MATLAB and Julia code. 

The file `run_experiments_NARMAX.m` generates NARMAX signals, writes them to file and runs the offline Iterative Least-Squares baseline. The notebook "run_experiments_NARMAX.ipynb" will read the generated signals and run the online Recursive Least-Squares baseline as well as our Free Energy Minimisation estimator. The file 'experiments_NARMAX.jl" contains functions to perform the experiments and `visualization.jl` speaks for itself. The notebook `collect-results.ipynb` collects all generated result files and plots the figures in the paper.