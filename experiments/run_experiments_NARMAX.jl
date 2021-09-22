using Revise
using JLD
using MAT
using ProgressMeter
using LinearAlgebra

using ForneyLab
import ForneyLab: unsafeMean, unsafeCov, unsafePrecision
using NARMAX

include("experiments_NARMAX.jl")
include("util.jl")


"""System parameters"""

# Polynomial degrees
deg_t = 3

# Orders
M1_t = 1
M2_t = 1
M3_t = 1
M_t = 1 + M1_t + M2_t + M3_t
N_t = M_t*deg_t

# Input signal params
num_periods = 1
points_period = 2^16
fMin = 0.0
fMax = 100.0
fs = 1000
uStd = 1.

# Output signal params
τ_true = 1e4
θ_scale = 0.5

# Basis function of system
options = Dict()
options["na"] = M1_t
options["nb"] = M2_t
options["ne"] = M3_t
options["nd"] = deg_t
options["dc"] = false
options["crossTerms"] = true
options["noiseCrossTerms"] = false
PΨ = gen_combs(options)
ψ(x::Array{Float64,1}) = [prod(x.^PΨ[:,k]) for k = 1:size(PΨ,2)]

"""Model parameters"""

# Polynomial degrees
deg_m = 3

# Orders
M1_m = 1
M2_m = 1
M3_m = 1
M_m = 1 + M1_m + M2_m + M3_m

# Basis function model
options = Dict()
options["na"] = M1_m
options["nb"] = M2_m
options["ne"] = M3_m
options["nd"] = deg_m
options["dc"] = false
options["crossTerms"] = true
options["noiseCrossTerms"] = true
PΦ = gen_combs(options)
ϕ(x::Array{Float64,1}) = [prod(x.^PΦ[:,k]) for k = 1:size(PΦ,2)]
N_m = size(PΦ,2)

# Initialize priors
priors = Dict("θ" => (zeros(N_m,), 1. .*Matrix{Float64}(I,N_m,N_m)), 
              "τ" => (1e2, 1e0))

# RLS forgetting factor
λ = 1.00

"""Experimental parameters"""

# Series of train sizes
trn_sizes = 2 .^collect(6:14)
num_trnsizes = length(trn_sizes)

# Define transient and test indices
transient = 0
ix_tst = collect(1:1000) .+ transient

# Number of VMP iterations
num_iters = 5

# Number of repetitions
num_repeats = 100

"""Run experiments"""

# Preallocate results arrays
results_sim_FEM = zeros(num_repeats, num_trnsizes)
results_prd_FEM = zeros(num_repeats, num_trnsizes)
results_sim_RLS = zeros(num_repeats, num_trnsizes)
results_prd_RLS = zeros(num_repeats, num_trnsizes)

# Specify model and compile update functions
source_code = model_specification(ϕ, M1=M1_m, M2=M2_m, M3=M3_m, N=N_m)
eval(Meta.parse(source_code))

println("Starting experiments..")
@showprogress for r = 1:num_repeats

    RMS_sim_FEM = zeros(num_trnsizes,)
    RMS_prd_FEM = zeros(num_trnsizes,)
    RMS_sim_RLS = zeros(num_trnsizes,)
    RMS_prd_RLS = zeros(num_trnsizes,)
    
    # Generate a signal
    # input, output, ix_trn, ix_val = generate_data(ψ, θ_scale=θ_scale, τ_true=τ_true, degree=deg_t, M1=M1_t, M2=M2_t, M3=M3_t, fMin=fMin, fMax=fMax, fs=fs, uStd=uStd, T=time_horizon, split_index=split_index, start_index=start_index, num_periods=num_periods, points_period=points_period)
    
    # Read from Maarten's code
    mat_data = matread("data/NARMAXsignal_order"*string(M_m)*"_N"*string(N_m)*"_r"*string(r)*".mat")

    for n = 1:num_trnsizes

        # Establish length of training signal
        ix_trn  = collect(1:trn_sizes[n]) .+ transient

        input_trn = mat_data["uTrain"][ix_trn]
        input_tst = mat_data["uTest"][ix_tst]
        output_trn = mat_data["yTrain"][ix_trn]
        output_tst = mat_data["yTest"][ix_tst]

        # Experiments        
        RMS_sim_FEM[n], RMS_prd_FEM[n] = experiment_FEM(input_trn, output_trn, input_tst, output_tst, ϕ, priors, M1=M1_m, M2=M2_m, M3=M3_m, N=N_m, num_iters=num_iters, computeFE=false)
        RMS_sim_RLS[n], RMS_prd_RLS[n] = experiment_RLS(input_trn, output_trn, input_tst, output_tst, ϕ, M1=M1_m, M2=M2_m, M3=M3_m, N=N_m, λ=λ)
    end

    # Write results to file
    save("results/results-NARMAX_FEM_M"*string(M_m)*"_N"*string(N_m)*"_degree"*string(deg_m)*"_r"*string(r)*".jld", "RMS_sim", RMS_sim_FEM, "RMS_prd", RMS_prd_FEM)
    save("results/results-NARMAX_RLS_M"*string(M_m)*"_N"*string(N_m)*"_degree"*string(deg_m)*"_r"*string(r)*".jld", "RMS_sim", RMS_sim_RLS, "RMS_prd", RMS_prd_RLS)

    results_prd_FEM[r,:] = RMS_prd_FEM
    results_sim_FEM[r,:] = RMS_sim_FEM
    results_prd_RLS[r,:] = RMS_prd_RLS
    results_sim_RLS[r,:] = RMS_sim_RLS

end

# Write results to file
println("Writing results to file..")
save("results/results-NARMAX_FEM_M"*string(M_m)*"_N"*string(N_m)*"_degree"*string(deg_m)*".jld", "RMS_sim", results_sim_FEM, "RMS_prd", results_prd_FEM)
save("results/results-NARMAX_RLS_M"*string(M_m)*"_N"*string(N_m)*"_degree"*string(deg_m)*".jld", "RMS_sim", results_sim_RLS, "RMS_prd", results_prd_RLS)

println("Experiments complete.")