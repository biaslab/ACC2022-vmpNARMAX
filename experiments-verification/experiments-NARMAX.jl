using JLD
using ProgressMeter
using LinearAlgebra
using ForneyLab
using NARMAX

import ForneyLab: unsafeMean, unsafeCov
include("gen_data.jl")

vis = false
if vis; include("visualization.jl"); end


function generate_data(ϕ; θ_scale=0.1, τ_true=1e3, degree=1, M1=1, M2=1, M3=1, fMin=0.8, fMax=1.0, fs=1.0, uStd=1., T=100, split_index=50, start_index=10, num_periods=num_periods, points_period=points_period)

    # Generate signal
    input, output = gensignalNARMAX(ϕ, θ_scale, τ_true; M1=M1, M2=M2, M3=M3, degree=degree, fMin=fMin, fMax=fMax, fs=fs, uStd=uStd, T=T, num_periods=num_periods, points_period=points_period)

    # Split data
    ix_trn, ix_val = split_data(T, split_index, start_index=start_index)

    return input, output, ix_trn, ix_val
end

function model_specification(ϕ; M1=M1, M2=M2, M3=M3, M=M)

    graph = FactorGraph()

    # Observed variables
    @RV x_kmin1; placeholder(x_kmin1, :x_kmin1, dims=(M1,))
    @RV z_kmin1; placeholder(z_kmin1, :z_kmin1, dims=(M2,))
    @RV r_kmin1; placeholder(r_kmin1, :r_kmin1, dims=(M3,))
    @RV u_k; placeholder(u_k, :u_k)

    # Time-invariant parameters
    @RV θ ~ GaussianMeanVariance(placeholder(:m_θ, dims=(M,)), placeholder(:v_θ, dims=(M,M)))
    @RV τ ~ Gamma(placeholder(:a_τ), placeholder(:b_τ))

    # Likelihood
    @RV y_k ~ NAutoRegressiveMovingAverageX(θ, x_kmin1, u_k, z_kmin1, r_kmin1, τ, g=ϕ)
    placeholder(y_k, :y_k)

    q = PosteriorFactorization(θ, τ, ids=[:θ :τ])
    algorithm = messagePassingAlgorithm([θ; τ], q, free_energy=true)
    return algorithmSourceCode(algorithm, free_energy=true)
    
end

function experiment_FEM(input, output, ix_trn, ix_val, ϕ; M1=1, M2=1, M3=1, N=3, T=100, num_iters=5, vis=false, computeFE=false)

    # Signal splits
    T_trn = length(ix_trn)
    T_val = length(ix_val)

    "Inference execution"

    # Initialize priors
    θ_k = (zeros(N,), 10 .*Matrix{Float64}(I,N,N))
    τ_k = (1e5, 1e0)

    # Initialize marginals
    marginals = Dict(:θ => ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=θ_k[1], v=θ_k[2]),
                     :τ => ProbabilityDistribution(Univariate, Gamma, a=τ_k[1], b=τ_k[2]))

    # Preallocate free energy array
    Fq = zeros(T_trn,num_iters)

    # Preallocate parameter arrays
    params_θ = (zeros(T_trn,N), zeros(T_trn,N,N))
    params_τ = (zeros(T_trn,1), zeros(T_trn,1))

    # Preallocate prediction arrays
    predictions = (zeros(T,), zeros(T,))
    residuals = zeros(T,)

    for (ii,k) in enumerate(ix_trn)
        
        # Update history vectors
        x_kmin1 = output[k-1:-1:k-M1]
        z_kmin1 = input[k-1:-1:k-M2]
        r_kmin1 = residuals[k-1:-1:k-M3]
        
        ϕx = ϕ([x_kmin1; input[k]; z_kmin1; r_kmin1])
        predictions[1][k] = θ_k[1]'*ϕx
        predictions[2][k] = ϕx'*θ_k[2]'*ϕx + inv(τ_k[1]/τ_k[2])

        # Compute prediction error
        residuals[k] = output[k] - predictions[1][k]
        
        # Set data 
        data = Dict(:y_k => output[k],
                    :u_k => input[k],
                    :x_kmin1 => x_kmin1,
                    :z_kmin1 => z_kmin1,
                    :r_kmin1 => r_kmin1,
                    :m_θ => θ_k[1],
                    :v_θ => θ_k[2],
                    :a_τ => τ_k[1],
                    :b_τ => τ_k[2])
        
        # Iterate updates
        for n = 1:num_iters
            
            if computeFE
                # Compute free energy
                Fq[ii,n] = freeEnergy(data,marginals)
            end
            
            # Update parameters
            stepθ!(data, marginals)
            stepτ!(data, marginals)
            
        end 
        
        # Update params
        θ_k = (unsafeMean(marginals[:θ]), unsafeCov(marginals[:θ]))
        τ_k = (marginals[:τ].params[:a], marginals[:τ].params[:b])
        
        # Store params
        params_θ[1][ii,:] = θ_k[1]
        params_θ[2][ii,:,:] = θ_k[2]
        params_τ[1][ii] = τ_k[1]
        params_τ[2][ii] = τ_k[2]
        
    end

    "Simulation"

    # Prepare array
    simulations = (zeros(T,), zeros(T,))
    simulations[1][ix_trn] = output[ix_trn]

    for k in ix_val
        
        # Update history vectors
        x_kmin1 = simulations[1][k-1:-1:k-M1]
        z_kmin1 = input[k-1:-1:k-M2]
        r_kmin1 = zeros(M3,)
            
        # Posterior predictive
        ϕx = ϕ([x_kmin1; input[k]; z_kmin1; r_kmin1])
        simulations[1][k] = θ_k[1]'*ϕx
        simulations[2][k] = ϕx'*θ_k[2]*ϕx + inv(τ_k[1] / τ_k[2])
        
    end

    "Evaluation"

    # Compute prediction errors
    sq_errors = (simulations[1][ix_val] - output[ix_val]).^2

    # Compute root mean square error
    RMS = sqrt(mean(sq_errors))

    if vis
        p1 = plot_forecast(output, simulations, ix_trn, ix_val, posterior=true)
        p2 = plot_errors(output, simulations[1], ix_val)

        savefig(p1, "figures/NARMAX-FEM_forecast.png")
        savefig(p2, "figures/NARMAX-FEM_errors.png")
    end

    return RMS, Fq
end

function experiment_RLS(input, output, ix_trn, ix_val, ϕ; M1=1, M2=1, M3=1, N=1, λ=1.00, T=100, vis=false)

    T_trn = length(ix_trn)

    # Parameters
    P = λ.*Matrix{Float64}(I,N,N)
    w_k = zeros(N,)

    # Preallocate prediction array
    predictions = zeros(T,)
    residuals = zeros(T,)

    for k in ix_trn
        
        # Update data vector
        ϕx = ϕ([output[k-1:-1:k-M1]; input[k:-1:k-M2]; residuals[k-1:-1:k-M3]])
        
        # Update weights
        α = output[k] - w_k'*ϕx 
        g = P*ϕx*inv(λ + ϕx'*P*ϕx)
        P = inv(λ)*P - g*ϕx'*inv(λ)*P
        w_k = w_k + α*g
        
        # Prediction
        predictions[k] = w_k'*ϕx
        residuals[k] = output[k] - predictions[k]
        
    end

    simulations = zeros(T,)
    simulations[ix_trn] = output[ix_trn]

    # Simulation
    for k in ix_val
        
        # Update data vector
        x = [simulations[k-1:-1:k-M1]; input[k:-1:k-M2]; zeros(M3,)]
        
        # Prediction
        simulations[k] = w_k'*ϕ(x)
        
    end

    # Compute prediction errors
    sq_errors = (simulations[ix_val] - output[ix_val]).^2

    # Compute root mean square error
    RMS = sqrt(mean(sq_errors))
    
    if vis
        p1 = plot_forecast(output, simulations, ix_trn, ix_val, posterior=false)
        p2 = plot_errors(output, simulations, ix_val)

        savefig(p1, "figures/NARX-RLS_forecast.png")
        savefig(p2, "figures/NARX-RLS_errors.png")
    end

    return RMS
end

"""Run experiments"""

# Polynomial degrees
deg_t = 3
deg_m = 3

# Orders
M1_t = 3
M2_t = 3
M3_t = 3
M_t = M1_t + 1 + M2_t + M3_t
N_t = M_t*deg_t + 1

M1_m = 1
M2_m = 1
M3_m = 1
M_m = M1_m + 1 + M2_m + M3_m
N_m = M_m*deg_m + 1

# Input signal params
num_periods = 10
points_period = 1000
fMin = 0.0
fMax = 100.0
fs = 10 .*fMax
uStd = 0.1

# Output signal params
λ = 1.00
τ_true = 1e6
θ_scale = 0.5

# Number of VMP iterations
num_iters = 5

# Basis function true signal
PΨ = zeros(M_t,1)
for d=1:deg_t; global PΨ = hcat(d .*Matrix{Float64}(I,M_t,M_t), PΨ); end
ψ(x::Array{Float64,1}) = [prod(x.^PΨ[:,k]) for k = 1:size(PΨ,2)]

# Basis function model
PΦ = zeros(M_m,1)
for d=1:deg_m; global PΦ = hcat(d .*Matrix{Float64}(I,M_m,M_m), PΦ); end
ϕ(x::Array{Float64,1}) = [prod(x.^PΦ[:,k]) for k = 1:size(PΦ,2)]

# Signal lengths
start_index = 50
split_index = 100 + start_index
time_horizon = 1000 + split_index

# Repetitions
num_repeats = 100
results_FEM = zeros(num_repeats,)
results_RLS = zeros(num_repeats,)

# Free energy
computeFE = true
FE = zeros(split_index-start_index+1, num_iters, num_repeats)

# Specify model and compile update functions
source_code = model_specification(ϕ, M1=M1_m, M2=M2_m, M3=M3_m, M=N_m)
eval(Meta.parse(source_code))

@showprogress for r = 1:num_repeats
    
    # Generate a signal
    input, output, ix_trn, ix_val = generate_data(ψ, θ_scale=θ_scale, τ_true=τ_true, degree=deg_t, M1=M1_t, M2=M2_t, M3=M3_t, fMin=fMin, fMax=fMax, fs=fs, uStd=uStd, T=time_horizon, split_index=split_index, start_index=start_index, num_periods=num_periods, points_period=points_period)

    # Experiments with different estimators
    results_FEM[r], FE[:,:,r] = experiment_FEM(input, output, ix_trn, ix_val, ϕ, M1=M1_m, M2=M2_m, M3=M3_m, N=N_m, T=time_horizon, vis=vis, computeFE=computeFE)
    results_RLS[r] = experiment_RLS(input, output, ix_trn, ix_val, ϕ, M1=M1_m, M2=M2_m, M3=M3_m, N=N_m, T=time_horizon, vis=vis, λ=λ)

end

# Report
println("Mean RMS FEM = "*string(mean(filter(!isinf, filter(!isnan, results_FEM))))*" ("*string(length(filter(isnan, results_FEM))/num_repeats)*"% rejected)")
println("Mean RMS RLS = "*string(mean(filter(!isinf, filter(!isnan, results_RLS))))*" ("*string(length(filter(isnan, results_RLS))/num_repeats)*"% rejected)")
println("Proportion FEM < RLS = "*string(mean(results_FEM .< results_RLS)))

# Write results to file
save("results/results-NARMAX_FEM_M"*string(M_m)*"_degree"*string(deg_m)*"_S"*string(split_index-start_index)*".jld", "RMS", results_FEM, "FE", FE)
save("results/results-NARMAX_RLS_M"*string(M_m)*"_degree"*string(deg_m)*"_S"*string(split_index-start_index)*".jld", "RMS", results_RLS)