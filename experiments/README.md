## Verification experiments

The Monte Carlo experiments are implemented in a combination of MATLAB and Julia code.

The file `run_experiments_NARMAX.m` generates NARMAX signals, writes them to file and runs the offline Iterative Least-Squares baseline. The notebook "run_experiments_NARMAX.ipynb" will read the generated signals and run the online Recursive Least-Squares baseline as well as our Free Energy Minimisation estimator. 

The folder `../algorithms` contains parameter estimators for polynomial NARMAX models. The script `util.jl` contains some utility functions.

The notebooks `visualize-*.ipynb` collect all generated results and plot the figures of the paper. The script `visualization.jl` contains helper functions.
