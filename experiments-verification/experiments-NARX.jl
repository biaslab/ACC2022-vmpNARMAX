using JLD
using ProgressMeter
using LinearAlgebra
using ForneyLab
using NARX

import ForneyLab: unsafeMean, unsafeCov
include("gen_data.jl")

vis = false
if vis; include("visualization.jl"); end


function generate_data(ϕ; θ_scale=0.1, τ_true=1e3, degree=1, M1=1, M2=1, T=100, split_index=50, start_index=10)

    # Generate signal
    input, output = gensignalNARX(ϕ, θ_scale, τ_true; M1=M1, M2=M2, degree=degree, T=T)

    # Split data
    ix_trn, ix_val = split_data(T, split_index, start_index=start_index)

    return input, output, ix_trn, ix_val
end

function model_specification(ϕ; M1=M1, M2=M2, M=M)

    graph = FactorGraph()

    # Observed variables
    @RV x_kmin1; placeholder(x_kmin1, :x_kmin1, dims=(M1,))
    @RV z_kmin1; placeholder(z_kmin1, :z_kmin1, dims=(M2,))
    @RV u_k; placeholder(u_k, :u_k)

    # Time-invariant parameters
    @RV θ ~ GaussianMeanVariance(placeholder(:m_θ, dims=(M,)), placeholder(:v_θ, dims=(M,M)))
    @RV τ ~ Gamma(placeholder(:a_τ), placeholder(:b_τ))

    # Likelihood
    @RV y_k ~ NAutoregressiveX(θ, x_kmin1, u_k, z_kmin1, τ, g=ϕ)
    placeholder(y_k, :y_k)

    q = PosteriorFactorization(θ, τ, ids=[:θ :τ])
    algorithm = messagePassingAlgorithm([θ; τ], q)
    return algorithmSourceCode(algorithm)
    
end

function experiment_FEM(input, output, ix_trn, ix_val, ϕ; M1=1, M2=1, M=3, T=100, vis=false)

    T_trn = length(ix_trn)
    T_val = length(ix_val)

    "Inference execution"

    # Preallocate parameter arrays
    params_θ = (zeros(T_trn,M), zeros(T_trn,M,M))
    params_τ = (zeros(T_trn,1), zeros(T_trn,1))

    # Initialize priors
    θ_k = (zeros(M,), 10 .*Matrix{Float64}(I,M,M))
    τ_k = (1e3, 1e0)

    # Initialize marginals
    marginals = Dict(:θ => ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=θ_k[1], v=θ_k[2]),
                     :τ => ProbabilityDistribution(Univariate, Gamma, a=τ_k[1], b=τ_k[2]))

    # Initialize prediction arrays
    predictions = (zeros(T,), zeros(T,))

    for k in ix_trn
        
        # Update history vectors
        x_kmin1 = output[k-1:-1:k-M1]
        z_kmin1 = input[k-1:-1:k-M2]
        
        ϕx = ϕ([x_kmin1; input[k]; z_kmin1])
        predictions[1][k] = θ_k[1]'*ϕx
        predictions[2][k] = ϕx'*θ_k[2]*ϕx + inv(τ_k[1] / τ_k[2])
        
        # Set data 
        data = Dict(:y_k => output[k],
                    :u_k => input[k],
                    :x_kmin1 => x_kmin1,
                    :z_kmin1 => z_kmin1,
                    :m_θ => θ_k[1],
                    :v_θ => θ_k[2],
                    :a_τ => τ_k[1],
                    :b_τ => τ_k[2])
        
        # Iterate updates
        for n = 1:10
            stepθ!(data, marginals)
            stepτ!(data, marginals)
        end    
        
        # Update params
        θ_k = (unsafeMean(marginals[:θ]), unsafeCov(marginals[:θ]))
        τ_k = (marginals[:τ].params[:a], marginals[:τ].params[:b])
        
        # Store params
        params_θ[1][k-ix_trn[1]+1,:] = θ_k[1]
        params_θ[2][k-ix_trn[1]+1,:,:] = θ_k[2]
        params_τ[1][k-ix_trn[1]+1] = τ_k[1]
        params_τ[2][k-ix_trn[1]+1] = τ_k[2]
        
    end

    "Simulation"

    for k in ix_val
        
        # Update history vectors
        x_kmin1 = predictions[1][k-1:-1:k-M1]
        z_kmin1 = input[k-1:-1:k-M2]
            
        # Posterior predictive
        ϕx = ϕ([x_kmin1; input[k]; z_kmin1])
        predictions[1][k] = θ_k[1]'*ϕx
        predictions[2][k] = ϕx'*θ_k[2]*ϕx + inv(τ_k[1] / τ_k[2])
        
    end

    "Evaluation"

    # Compute prediction errors
    sq_errors = (predictions[1][ix_val] - output[ix_val]).^2

    # Compute root mean square error
    RMS = sqrt(mean(sq_errors))

    if vis
        p1 = plot_forecast(output, predictions, ix_trn, ix_val, posterior=true)
        p2 = plot_errors(output, predictions[1], ix_val)

        savefig(p1, "figures/NARX-FEM_forecast.png")
        savefig(p2, "figures/NARX-FEM_errors.png")
    end

    return RMS
end

function experiment_RLS(input, output, ix_trn, ix_val, ϕ; M1=M1, M2=M2, M=M, λ=1.00, T=100, vis=false)

    T_trn = length(ix_trn)

    # Parameters
    P = λ.*Matrix{Float64}(I,M,M)
    w_k = zeros(M,)

    # Preallocate prediction array
    predictions = zeros(T,)

    for k in ix_trn
        
        # Update data vector
        ϕx = ϕ([output[k-1:-1:k-M1]; input[k:-1:k-M2]])
        
        # Update weights
        α = output[k] - w_k'*ϕx 
        g = P*ϕx*inv(λ + ϕx'*P*ϕx)
        P = inv(λ)*P - g*ϕx'*inv(λ)*P
        w_k = w_k + α*g
        
        # Prediction
        predictions[k] = w_k'*ϕx
        
    end

    # Simulation
    for k in ix_val
        
        # Update data vector
        x = [predictions[k-1:-1:k-M1]; input[k:-1:k-M2]]
        
        # Prediction
        predictions[k] = w_k'*ϕ(x)
        
    end

    # Compute prediction errors
    sq_errors = (predictions[ix_val] - output[ix_val]).^2

    # Compute root mean square error
    RMS = sqrt(mean(sq_errors))
    
    if vis
        p1 = plot_forecast(output, predictions, ix_trn, ix_val, posterior=false)
        p2 = plot_errors(output, predictions, ix_val)

        savefig(p1, "figures/NARX-RLS_forecast.png")
        savefig(p2, "figures/NARX-RLS_errors.png")
    end

    return RMS
end

# Orders
deg_true = 5
degree = 4
M1 = 2
M2 = 2
M = M1+1+M2
N = (M1+1+M2)*degree + 1

# Parameters
λ = 0.98
τ_true = 1e4
θ_scale = 0.5

# Signal lengths
start_index = 50
split_index = 50 + start_index
time_horizon = 1000 + split_index

# Basis functions
PΨ = zeros(M,1)
for d=1:deg_true; global PΨ = hcat(d .*Matrix{Float64}(I,M,M), PΨ); end
ψ(x::Array{Float64,1}) = [prod(x.^PΨ[:,k]) for k = 1:size(PΨ,2)]
PΦ = zeros(M,1)
for d=1:degree; global PΦ = hcat(d .*Matrix{Float64}(I,M,M), PΦ); end
ϕ(x::Array{Float64,1}) = [prod(x.^PΦ[:,k]) for k = 1:size(PΦ,2)]

# Repetitions
num_repeats = 100
results_FEM = zeros(num_repeats,)
results_RLS = zeros(num_repeats,)

# Specify model and compile update functions
source_code = model_specification(ϕ, M1=M1, M2=M2, M=N)
eval(Meta.parse(source_code))

@showprogress for r = 1:num_repeats
    
    # Generate a signal
    input, output, ix_trn, ix_val = generate_data(ψ, θ_scale=θ_scale, τ_true=τ_true, degree=deg_true, M1=M1, M2=M2, T=time_horizon, split_index=split_index, start_index=start_index)

    # Experiments with different estimators
    results_FEM[r] = experiment_FEM(input, output, ix_trn, ix_val, ϕ, M1=M1, M2=M2, M=N, T=time_horizon, vis=vis)
    results_RLS[r] = experiment_RLS(input, output, ix_trn, ix_val, ϕ, M1=M1, M2=M2, M=N, T=time_horizon, vis=vis, λ=λ)

end

# Report
println("Mean RMS FEM = "*string(mean(filter(!isinf, filter(!isnan, results_FEM)))))
println("Mean RMS RLS = "*string(mean(filter(!isinf, filter(!isnan, results_RLS)))))

# Write results to file
save("results/results-NARX_FEM_M"*string(M)*"_degree"*string(degree)*"_S"*string(split_index-start_index)*".jld", "RMS", results_FEM)
save("results/results-NARX_RLS_M"*string(M)*"_degree"*string(degree)*"_S"*string(split_index-start_index)*".jld", "RMS", results_RLS)