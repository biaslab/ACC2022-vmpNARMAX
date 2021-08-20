using Revise
using Rocket
using GraphPPL
using ReactiveMP # develop-narmax
using LinearAlgebra
using Plots
import ProgressMeter

# NARMAX model definition
@model [ default_factorisation = MeanField() ] function narmax_model(n, order, f)

    x = datavar(Vector{Float64}, n)
    u = datavar(Vector{Float64}, n)
    z = datavar(Vector{Float64}, n)
    r = datavar(Vector{Float64}, n)

    y = datavar(Float64, n)

    τ ~ GammaShapeRate(1.0, 1.0)
    θ ~ MvNormalMeanPrecision(zeros(order), Matrix{Float64}(I, order, order))

    meta = NARMAXMeta(f, nothing, nothing, TinyCorrection())

    for i in 1:n
        y[i] ~ NARMAX(θ, x[i], u[i], z[i], r[i], τ) where {meta=meta}
    end

    return y, θ, x, u, z, r, τ
end

# NARMAX inference
function inference_narmax(inputs_arr, outputs, order, niter, f)
    n = length(outputs)
    x_inputs, u_inputs, z_inputs, r_inputs  = inputs_arr
    model, (y, θ, x, u, z, r, τ) = narmax_model(n, order, f)
    
    τ_buffer = nothing
    θ_buffer = nothing
    fe = Vector{Float64}()
    # fe_scheduler = PendingScheduler()

    subscribe!(getmarginal(τ), (mτ) -> τ_buffer = mτ)
    subscribe!(getmarginal(θ), (mθ) -> θ_buffer = mθ)
    subscribe!(score(Float64, BetheFreeEnergy(), model), (f) -> push!(fe, f))

    setmarginal!(τ, GammaShapeRate(1.0, 1.0))
    setmarginal!(θ, MvNormalMeanPrecision(zeros(order), Matrix{Float64}(I, order, order)))

    ProgressMeter.@showprogress for i in 1:niter
        update!(x, x_inputs)
        update!(u, u_inputs)
        update!(z, z_inputs)
        update!(r, r_inputs)
        update!(y, outputs)
        # release!(fe_scheduler)
    end


    return τ_buffer, θ_buffer, fe
end

# FIXME: Generation code is not clean
include("../experiments-verification/gen-data/fMultiSinGen.jl")
function generate_data()
    # Polynomial degrees
    deg_t = 3

    # True orders
    M1_t = 3
    M2_t = 3
    M3_t = 3
    M_t = M1_t + 1 + M2_t + M3_t

    # Number of coefficients
    N_t = M_t*deg_t
    # N_t = M_t*deg_t + 1

    # True basis function
    PΨ = zeros(M_t,0); for d=1:deg_t; PΨ = hcat(d .*Matrix{Float64}(I,M_t,M_t), PΨ); end
    ψ(x::Array{Float64,1}) = [prod(x.^PΨ[:,k]) for k = 1:size(PΨ,2)];

    # Parameters
    τ_true = 1e6
    θ_true = .5 .*(rand(N_t,) .- 0.5);
    θ_true[end] = 0.;



    # Parameters
    num_periods = 10
    points_period = 1000
    num_real = 1
    fMin = 0.0
    fMax = 100.0
    fs = 10 .* fMax
    uStd = 0.1

    # Input frequency and amplitude
    input, inputfreq = fMultiSinGen(points_period, 
                                    num_periods, 
                                    num_real, 
                                    fMin=fMin, 
                                    fMax=fMax, 
                                    fs=fs, 
                                    type_signal="odd", 
                                    uStd=uStd);

    # Scale down
    # input /= 10.;
    T = length(input)
    # Observation array
    output = zeros(T,)
    errors = zeros(T,)

    for k = 1:T

        # Generate noise
        errors[k] = sqrt(inv(τ_true))*randn(1)[1]

        # Output
        if k < (maximum([M1_t, M2_t, M3_t])+1)
            output[k] = input[k] + errors[k]
        else
            # Update history vectors
            x_kmin1 = output[k-1:-1:k-M1_t]
            z_kmin1 = input[k-1:-1:k-M2_t]
            r_kmin1 = errors[k-1:-1:k-M3_t]

            # Compute output
            output[k] = θ_true'*ψ([x_kmin1; input[k]; z_kmin1; r_kmin1]) + errors[k]
        end
    end
    input, output, θ_true, τ_true
end

# FIXME: I didn't get how to map inputs to vectors x, u, r
inputs, outputs, theta, tau = generate_data()
# Hence I generate dumb data
n_samples = 100
degree = 3
# generate x, u, z, r
inputs = [collect.(eachrow(randn(n_samples, degree))), collect.(eachrow(randn(n_samples, degree))), collect.(eachrow(randn(n_samples, degree))), collect.(eachrow(randn(n_samples, 1)))]
outputs = randn(n_samples)

# FIXME: Need comments, what is gen_combs?
# Autoregression orders
M1 = 3
M2 = 3
M3 = 3
M = 1 + M1 + M2 + M3
# Basis expansion
degree = 3
options = Dict()
options["na"] = M1
options["nb"] = M2
options["ne"] = M3
options["nd"] = degree
options["dc"] = true
options["crossTerms"] = false
options["noiseCrossTerms"] = false
# FIXME: 
include("../experiments-verification/util.jl")
PΦ = gen_combs(options)

ϕ(x::Array{Float64,1}) = [prod(x.^PΦ[:,k]) for k = 1:size(PΦ,2)];

# INFERENCE
τ_res, θ_res, fe_res = inference_narmax(inputs, outputs, 98, 10, ϕ)

# RESULT
mean(τ_res)
mean_cov(θ_res)
plot(fe_res)