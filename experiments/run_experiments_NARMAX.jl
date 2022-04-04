using Revise
using JLD
using MAT
using ProgressMeter
using LinearAlgebra
using ForneyLab
using NARMAX

include("../algorithms/VMP-estimator-NARMAX.jl")
include("../algorithms/RLS-estimator-NARMAX.jl")
include("util.jl")


"""Experimental parameters"""

# System noise std deviation
stde = 0.03

# Series of train sizes
trn_sizes = 2 .^collect(7:10)
num_trnsizes = length(trn_sizes)

# Define transient and test indices
transient = 0
ix_tst = collect(1:1000) .+ transient

# Number of VMP iterations
num_iters = 10

# Number of repetitions
num_repeats = 100

"""Model parameters"""

# Polynomial degree
degree = 3

# Orders
M1 = 1
M2 = 1
M3 = 1
M = 1 + M1 + M2 + M3

# Basis function model
options = Dict()
options["na"] = M1
options["nb"] = M2
options["ne"] = M3
options["nd"] = degree
options["dc"] = false
options["crossTerms"] = true
options["noiseCrossTerms"] = false
PΦ = gen_combs(options)
ϕ(x::Array{Float64,1}) = [prod(x.^PΦ[:,k]) for k = 1:size(PΦ,2)]
N = size(PΦ,2)

# Initialize priors
priors = Dict("θ" => (zeros(N,), Matrix{Float64}(I,N,N)), 
              "τ" => (10, 0.004))
prior_mτ = priors["τ"][1] ./ priors["τ"][2]              

# RLS forgetting factor
λ = 1.00

"""Run experiments"""

# Preallocate results arrays
results_sim_VMP = zeros(num_repeats, num_trnsizes)
results_prd_VMP = zeros(num_repeats, num_trnsizes)
results_sim_RLS = zeros(num_repeats, num_trnsizes)
results_prd_RLS = zeros(num_repeats, num_trnsizes)
results_sim_SYS = zeros(num_repeats, num_trnsizes)
results_prd_SYS = zeros(num_repeats, num_trnsizes)

# Specify model and compile update functions
source_code = model_specification(ϕ, M1=M1, M2=M2, M3=M3, N=N)
eval(Meta.parse(source_code))

println("Starting experiments..")
@showprogress for r = 1:num_repeats

    RMS_sim_VMP = zeros(num_trnsizes,)
    RMS_prd_VMP = zeros(num_trnsizes,)
    RMS_sim_RLS = zeros(num_trnsizes,)
    RMS_prd_RLS = zeros(num_trnsizes,)
    RMS_sim_SYS = zeros(num_trnsizes,)
    RMS_prd_SYS = zeros(num_trnsizes,)
    
    # Read matlab-generated signal
    mat_data = matread("data/NARMAXsignal_stde"*string(stde)*"_pol3_order"*string(M)*"_N"*string(N)*"_r"*string(r)*".mat")

    for n = 1:num_trnsizes

        # Split signals
        ix_trn  = collect(1:trn_sizes[n]) .+ transient
        input_trn = mat_data["uTrain"][ix_trn]
        input_tst = mat_data["uTest"][ix_tst]
        output_trn = mat_data["yTrain"][ix_trn]
        output_tst = mat_data["yTest"][ix_tst]

        # System parameters
        θ_sys = mat_data["system"]["theta"][:]
        τ_sys = inv(mat_data["options"]["stde"]^2)

        # Experiments        
        RMS_sim_VMP[n], RMS_prd_VMP[n] = VMP(input_trn, output_trn, input_tst, output_tst, ϕ, priors, M1=M1, M2=M2, M3=M3, N=N, num_iters=num_iters, computeFE=false)
        RMS_sim_RLS[n], RMS_prd_RLS[n] = RLS(input_trn, output_trn, input_tst, output_tst, ϕ, M1=M1, M2=M2, M3=M3, N=N, λ=λ)
        RMS_sim_SYS[n], RMS_prd_SYS[n] = run_system(input_tst, output_tst, ϕ, θ_sys, M1=M1, M2=M2, M3=M3, N=N)
    end

    # Write results to file
    save("results/results-NARMAX_VMP_stde"*string(stde)*"_pol3_M"*string(M)*"_N"*string(N)*"_degree"*string(degree)*"_mtau_"*string(prior_mτ)*"_r"*string(r)*".jld", "RMS_sim", RMS_sim_VMP, "RMS_prd", RMS_prd_VMP)
    save("results/results-NARMAX_RLS_stde"*string(stde)*"_pol3_M"*string(M)*"_N"*string(N)*"_degree"*string(degree)*"_r"*string(r)*".jld", "RMS_sim", RMS_sim_RLS, "RMS_prd", RMS_prd_RLS)
    save("results/results-NARMAX_SYS_stde"*string(stde)*"_pol3_M"*string(M)*"_N"*string(N)*"_degree"*string(degree)*"_r"*string(r)*".jld", "RMS_sim", RMS_sim_SYS, "RMS_prd", RMS_prd_SYS)

    results_prd_VMP[r,:] = RMS_prd_VMP
    results_sim_VMP[r,:] = RMS_sim_VMP
    results_prd_RLS[r,:] = RMS_prd_RLS
    results_sim_RLS[r,:] = RMS_sim_RLS
    results_prd_SYS[r,:] = RMS_prd_SYS
    results_sim_SYS[r,:] = RMS_sim_SYS

end

# Write results to file
println("Writing results to file..")
save("results/results-NARMAX_VMP_stde"*string(stde)*"_pol3_M"*string(M)*"_N"*string(N)*"_degree"*string(degree)*".jld", "RMS_sim", results_sim_VMP, "RMS_prd", results_prd_VMP)
save("results/results-NARMAX_RLS_stde"*string(stde)*"_pol3_M"*string(M)*"_N"*string(N)*"_degree"*string(degree)*".jld", "RMS_sim", results_sim_RLS, "RMS_prd", results_prd_RLS)
save("results/results-NARMAX_SYS_stde"*string(stde)*"_pol3_M"*string(M)*"_N"*string(N)*"_degree"*string(degree)*".jld", "RMS_sim", results_sim_RLS, "RMS_prd", results_prd_RLS)

println("Experiments complete.")