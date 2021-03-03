using JLD
using MAT
using ProgressMeter
using LinearAlgebra

using ForneyLab
import ForneyLab: unsafeMean, unsafeCov
using NARMAX

include("gen_signal.jl")
include("experiments-NARMAX.jl")


# Polynomial degrees
deg_t = 3
deg_m = 3

# Orders
M1_t = 3
M2_t = 3
M3_t = 3
M_t = M1_t + 1 + M2_t + M3_t
N_t = M_t*deg_t + 1

M1_m = 3
M2_m = 3
M3_m = 3
M_m = M1_m + 1 + M2_m + M3_m
N_m = M_m*deg_m + 1

# Input signal params
num_periods = 10
points_period = 1000
fMin = 0.0
fMax = 100.0
fs = 1000
uStd = 1.

# Output signal params
λ = 1.00
τ_true = 1e4
θ_scale = 0.5

# Number of VMP iterations
num_iters = 10

# Basis function true signal
PΨ = zeros(M_t,1)
for d=1:deg_t; global PΨ = hcat(d .*Matrix{Float64}(I,M_t,M_t), PΨ); end
ψ(x::Array{Float64,1}) = [prod(x.^PΨ[:,k]) for k = 1:size(PΨ,2)]

# Basis function model
PΦ = zeros(M_m,1)
for d=1:deg_m; global PΦ = hcat(d .*Matrix{Float64}(I,M_m,M_m), PΦ); end
ϕ(x::Array{Float64,1}) = [prod(x.^PΦ[:,k]) for k = 1:size(PΦ,2)]

# Signal lengths
start_index = 10
split_index = 200 + start_index
time_horizon = 1000 + split_index
ix_trn = 1:1000
ix_tst = 1:1000

# Repetitions
num_repeats = 10
RMS_sim_FEM = zeros(num_repeats,)
RMS_prd_FEM = zeros(num_repeats,)
RMS_sim_RLS = zeros(num_repeats,)
RMS_prd_RLS = zeros(num_repeats,)

# Free energy
computeFE = true
FE = zeros(length(ix_trn), num_iters, num_repeats)

# Specify model and compile update functions
source_code = model_specification(ϕ, M1=M1_m, M2=M2_m, M3=M3_m, M=N_m)
eval(Meta.parse(source_code))

@showprogress for r = 1:num_repeats
    
    # Generate a signal
    # input, output, ix_trn, ix_val = generate_data(ψ, θ_scale=θ_scale, τ_true=τ_true, degree=deg_t, M1=M1_t, M2=M2_t, M3=M3_t, fMin=fMin, fMax=fMax, fs=fs, uStd=uStd, T=time_horizon, split_index=split_index, start_index=start_index, num_periods=num_periods, points_period=points_period)
    
    # Read from Maarten's code
    mat_data = matread("data/NARMAXsignal_r"*string(r)*".mat")
    input_trn = mat_data["uTrain"][ix_trn]
    input_tst = mat_data["uTest"][ix_tst]
    output_trn = mat_data["yTrain"][ix_trn]
    output_tst = mat_data["yTest"][ix_tst]

    # Experiments with different estimators
    RMS_sim_FEM[r], RMS_prd_FEM[r], FE[:,:,r] = experiment_FEM(input_trn, output_trn, input_tst, output_tst, ϕ, M1=M1_m, M2=M2_m, M3=M3_m, N=N_m, num_iters=num_iters, computeFE=computeFE)
    RMS_sim_RLS[r], RMS_prd_RLS[r] = experiment_RLS(input_trn, output_trn, input_tst, output_tst, ϕ, M1=M1_m, M2=M2_m, M3=M3_m, N=N_m, λ=λ)

end

# Report
println("Mean RMS prd FEM = "*string(mean(filter(!isinf, filter(!isnan, RMS_prd_FEM))))*" ("*string(length(filter(isnan, RMS_prd_FEM))/num_repeats)*"% rejected)")
println("Mean RMS sim FEM = "*string(mean(filter(!isinf, filter(!isnan, RMS_sim_FEM))))*" ("*string(length(filter(isnan, RMS_sim_FEM))/num_repeats)*"% rejected)")
println("Mean RMS prd RLS = "*string(mean(filter(!isinf, filter(!isnan, RMS_prd_RLS))))*" ("*string(length(filter(isnan, RMS_prd_RLS))/num_repeats)*"% rejected)")
println("Mean RMS sim RLS = "*string(mean(filter(!isinf, filter(!isnan, RMS_sim_RLS))))*" ("*string(length(filter(isnan, RMS_sim_RLS))/num_repeats)*"% rejected)")

# Write results to file
save("results/results-NARMAX_FEM_M"*string(M_m)*"_degree"*string(deg_m)*"_S"*string(length(ix_trn))*".jld", "RMS_sim", RMS_sim_FEM, "RMS_prd", RMS_prd_FEM, "FE", FE)
save("results/results-NARMAX_RLS_M"*string(M_m)*"_degree"*string(deg_m)*"_S"*string(length(ix_trn))*".jld", "RMS_sim", RMS_sim_RLS, "RMS_prd", RMS_prd_RLS)