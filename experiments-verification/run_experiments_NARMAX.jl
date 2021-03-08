using Revise
using JLD
using MAT
using ProgressMeter
using LinearAlgebra

using ForneyLab
import ForneyLab: unsafeMean, unsafeCov
using NARMAX

# include("gen_signal.jl")
include("experiments-NARMAX.jl")


"""System parameters"""

# Polynomial degrees
deg_t = 3

# Orders
M1_t = 3
M2_t = 3
M3_t = 3
M_t = M1_t + 1 + M2_t + M3_t
N_t = M_t*deg_t

# Input signal params
num_periods = 10
points_period = 1000
fMin = 0.0
fMax = 100.0
fs = 1000
uStd = 1.

# Output signal params
τ_true = 1e4
θ_scale = 0.5

# Basis function true signal
PΨ = zeros(M_t,0)
for d=1:deg_t; global PΨ = hcat(d .*Matrix{Float64}(I,M_t,M_t), PΨ); end
ψ(x::Array{Float64,1}) = [prod(x.^PΨ[:,k]) for k = 1:size(PΨ,2)]

"""Model parameters"""

# Polynomial degrees
deg_m = 3

# Orders
M1_m = 3
M2_m = 3
M3_m = 3
M_m = M1_m + 1 + M2_m + M3_m
N_m = M_m*deg_m

# Initialize priors
priors = Dict("θ" => (zeros(N_m,), 10 .*Matrix{Float64}(I,N_m,N_m)), 
              "τ" => (1e4, 1e0))

# RLS forgetting factor
λ = 1.00

# Basis function model
PΦ = zeros(M_m,0)
for d=1:deg_m; global PΦ = hcat(d .*Matrix{Float64}(I,M_m,M_m), PΦ); end
ϕ(x::Array{Float64,1}) = [prod(x.^PΦ[:,k]) for k = 1:size(PΦ,2)]

"""Experimental parameters"""

# Series of train sizes
trn_sizes = 2 .^collect(8:12)
# trn_sizes = [500, 1000, 2000, 4000]
# trn_sizes = [200, 400, 800, 1600]
num_trnsizes = length(trn_sizes)

# Define transient and test indices
transient = 0
ix_tst = collect(1:1000) .+ transient

# Number of VMP iterations
num_iters = 5

# Number of repetitions
num_repeats = 10

# Switch to compute FE
computeFE = true

"""Run experiments"""

# Preallocate results arrays
results_sim_FEM = zeros(num_repeats, num_trnsizes)
results_prd_FEM = zeros(num_repeats, num_trnsizes)
results_sim_RLS = zeros(num_repeats, num_trnsizes)
results_prd_RLS = zeros(num_repeats, num_trnsizes)

# Preallocate free energy arrays
avg_FE = zeros(num_repeats, num_trnsizes)
fin_FE = zeros(num_repeats, num_trnsizes)

# Specify model and compile update functions
source_code = model_specification(ϕ, M1=M1_m, M2=M2_m, M3=M3_m, M=N_m)
eval(Meta.parse(source_code))

@showprogress for r = 1:num_repeats

    RMS_sim_FEM = zeros(num_trnsizes,)
    RMS_prd_FEM = zeros(num_trnsizes,)
    RMS_sim_RLS = zeros(num_trnsizes,)
    RMS_prd_RLS = zeros(num_trnsizes,)

    avg_FE_r = zeros(num_trnsizes,)
    fin_FE_r = zeros(num_trnsizes,)
    
    # Generate a signal
    # input, output, ix_trn, ix_val = generate_data(ψ, θ_scale=θ_scale, τ_true=τ_true, degree=deg_t, M1=M1_t, M2=M2_t, M3=M3_t, fMin=fMin, fMax=fMax, fs=fs, uStd=uStd, T=time_horizon, split_index=split_index, start_index=start_index, num_periods=num_periods, points_period=points_period)
    
    # Read from Maarten's code
    mat_data = matread("data/NARMAXsignal_r"*string(r)*".mat")

    for n = 1:num_trnsizes

        # Establish length of training signal
        ix_trn = collect(1:trn_sizes[n]) .+ transient

        input_trn = mat_data["uTrain"][ix_trn]
        input_tst = mat_data["uTest"][ix_tst]
        output_trn = mat_data["yTrain"][ix_trn]
        output_tst = mat_data["yTest"][ix_tst]

        # Experiments
        try
            RMS_sim_FEM[n], RMS_prd_FEM[n], FE = experiment_FEM(input_trn, output_trn, input_tst, output_tst, ϕ, priors, M1=M1_m, M2=M2_m, M3=M3_m, N=N_m, num_iters=num_iters, computeFE=computeFE)
            avg_FE_r[n] = mean(FE[:,end])
            fin_FE_r[n] = FE[end,end]
        catch DomainError
            RMS_sim_FEM[n] = NaN
            RMS_prd_FEM[n] = NaN
            avg_FE_r[n] = NaN
            fin_FE_r[n] = NaN
        end

        RMS_sim_RLS[n], RMS_prd_RLS[n] = experiment_RLS(input_trn, output_trn, input_tst, output_tst, ϕ, M1=M1_m, M2=M2_m, M3=M3_m, N=N_m, λ=λ)
    end

    # Write results to file
    save("results/results-NARMAX_FEM_M"*string(M_m)*"_degree"*string(deg_m)*"_r"*string(r)*".jld", "RMS_sim", RMS_sim_FEM, "RMS_prd", RMS_prd_FEM, "avg_FE", avg_FE_r, "fin_FE", fin_FE_r)
    save("results/results-NARMAX_RLS_M"*string(M_m)*"_degree"*string(deg_m)*"_r"*string(r)*".jld", "RMS_sim", RMS_sim_RLS, "RMS_prd", RMS_prd_RLS)

    results_prd_FEM[r,:] = RMS_prd_FEM
    results_sim_FEM[r,:] = RMS_sim_FEM
    results_prd_RLS[r,:] = RMS_prd_RLS
    results_sim_RLS[r,:] = RMS_sim_RLS

    avg_FE[r,:] = avg_FE_r
    fin_FE[r,:] = fin_FE_r
end

# Write results to file
save("results/results-NARMAX_FEM_M"*string(M_m)*"_degree"*string(deg_m)*".jld", "RMS_sim", results_sim_FEM, "RMS_prd", results_prd_FEM, "avg_FE", avg_FE, "fin_FE", fin_FE)
save("results/results-NARMAX_RLS_M"*string(M_m)*"_degree"*string(deg_m)*".jld", "RMS_sim", results_sim_RLS, "RMS_prd", results_prd_RLS)

# Report
println("Mean RMS prd FEM = "*string(mean(filter(!isnan, results_prd_FEM)))*" ("*string(mean(isnan.(results_prd_FEM)))*"% rejected)")
println("Mean RMS sim FEM = "*string(mean(filter(!isnan, results_sim_FEM)))*" ("*string(mean(isnan.(results_sim_FEM)))*"% rejected)")
println("Mean RMS prd RLS = "*string(mean(filter(!isnan, results_prd_RLS)))*" ("*string(mean(isnan.(results_prd_FEM)))*"% rejected)")
println("Mean RMS sim RLS = "*string(mean(filter(!isnan, results_sim_RLS)))*" ("*string(mean(isnan.(results_sim_FEM)))*"% rejected)")
