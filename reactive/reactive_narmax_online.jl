using Revise
using Rocket
using ReactiveMP
using GraphPPL

# NARMAX model definition
@model [ default_factorisation = MeanField() ] function online_narmax_model(f)

    prior_τα = datavar(Float64)
    prior_τβ = datavar(Float64)
    prior_mθ = datavar(Vector{Float64})
    prior_vθ = datavar(Matrix{Float64})

    x = datavar(Vector{Float64})
    u = datavar(Vector{Float64})
    z = datavar(Vector{Float64})
    r = datavar(Vector{Float64})
    y = datavar(Float64)

    τ ~ GammaShapeRate(prior_τα, prior_τβ)
    θ ~ MvNormalMeanPrecision(prior_mθ, prior_vθ)

    meta = NARMAXMeta(f, nothing, nothing, TinyCorrection())

    y ~ NARMAX(θ, x, u, z, r, τ) where {meta=meta}

    return y, θ, x, u, z, r, τ, prior_τα, prior_τβ, prior_mθ, prior_vθ
end

function online_inference(inputs_arr, outputs, order, niter, f)
    n = length(outputs)
    x_inputs, u_inputs, z_inputs, r_inputs  = inputs_arr
    model, (y, θ, x, u, z, r, τ, prior_τα, prior_τβ, prior_mθ, prior_vθ) = online_narmax_model(f)
    
    ms_scheduler = PendingScheduler()
    fe_scheduler = PendingScheduler()
    
    τ_buffer = []
    θ_buffer = []
    fe = Vector{Float64}()

    sub_τ = subscribe!(getmarginal(τ) |> schedule_on(ms_scheduler), (m) -> push!(τ_buffer, m))
    sub_θ = subscribe!(getmarginal(θ) |> schedule_on(ms_scheduler), (m) -> push!(θ_buffer, m))
    sub_fe = subscribe!(score(BetheFreeEnergy(), model, fe_scheduler), (f) -> push!(fe, f))
    
    # Initial prior messages
    current_θ = MvNormalMeanPrecision(zeros(order), Matrix{Float64}(I, order, order))
    current_τ = GammaShapeRate(1.0, 1.0)

    # Initial marginals
    setmarginal!(τ, GammaShapeRate(1.0, 1.0))
    setmarginal!(θ, MvNormalMeanPrecision(zeros(order), Matrix{Float64}(I, order, order)))
    
    for i in 1:n
        
        for _ in 1:niter
            update!(x, x_inputs[i])
            update!(u, u_inputs[i])
            update!(z, z_inputs[i])
            update!(r, r_inputs[i])
            update!(y, outputs[i])

            update!(prior_τα, shape(current_τ))
            update!(prior_τβ, rate(current_τ))
            update!(prior_mθ, mean(current_θ))
            update!(prior_vθ, cov(current_θ))
            
            release!(fe_scheduler)
        end
        
        release!(ms_scheduler)
        
        current_θ = θ_buffer[end]
        current_τ = τ_buffer[end]
    end
    
    unsubscribe!(sub_τ)
    unsubscribe!(sub_θ)
    unsubscribe!(sub_fe)
    
    return θ_buffer, τ_buffer, fe
end

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

θ_res, τ_res, fe_res = online_inference(inputs, outputs, 98, 10, ϕ)

plot(sum(reshape(fe_res, (10, 100)), dims=2)./10)