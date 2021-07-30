# FREE ENERGY doesn't work in this model

using Plots
using ReactiveMP, Rocket, Distributions, GraphPPL, LinearAlgebra
import ProgressMeter
# NARMAX model definition
@model [default_factorisation=MeanField()] function narmax_w_future(f)
	prior_τα = datavar(Float64)
    prior_τβ = datavar(Float64)
    prior_mθ = datavar(Vector{Float64})
    prior_vθ = datavar(Matrix{Float64})

    x = datavar(Vector{Float64})
    u = datavar(Vector{Float64})
    z = datavar(Vector{Float64})
    r = datavar(Vector{Float64})
    y = datavar(Float64)

    τ ~ GammaShapeRate(prior_τα, prior_τβ) where {q = MeanField()}
    θ ~ MvNormalMeanPrecision(prior_mθ, prior_vθ) where {q = MeanField()}
	
	meta = NARMAXMeta(f, nothing, nothing, TinyCorrection())

	y ~ NARMAX(θ, x, u, z, r, τ) where {meta=meta}

	x_future = datavar(Vector{Float64})
	u_future = datavar(Vector{Float64})
	z_future = datavar(Vector{Float64})
	r_future = datavar(Vector{Float64})
	y_future = datavar(Float64)
	prediction, y_future ~ NARMAX(θ, x_future, u_future, z_future, r_future, τ) where { 
		q = MeanField(), meta=meta
	}
    return y, θ, x, u, z, r, τ, 
		   prior_τα, prior_τβ, 
		   prior_mθ, prior_vθ,
		   x_future, u_future, 
		   z_future, r_future, 
		   y_future, prediction
end

# AR inference
function infer_predict_narmax(inputs_arr, outputs, order, niter, f)

	x_inputs, u_inputs, z_inputs, r_inputs  = inputs_arr
    n = length(outputs)
    model, (y, θ, x, u, z, r, τ, prior_τα, prior_τβ, prior_mθ, prior_vθ, x_future, u_future, z_future, r_future, y_future, prediction) = narmax_w_future(f)

	m_scheduler = PendingScheduler()
	f_scheduler = PendingScheduler()
	p_buffer = keep(Message)
    τ_buffer = keep(Marginal)
    θ_buffer = keep(Marginal)
    fe       = ScoreActor(Float64)
    τsub = subscribe!(getmarginal(τ) |> schedule_on(m_scheduler), τ_buffer)
    θsub = subscribe!(getmarginal(θ) |> schedule_on(m_scheduler), θ_buffer)
    # fesub = subscribe!(score(Float64, BetheFreeEnergy(), model, f_scheduler), fe)
	predictive_messages = messageout(first(interfaces(prediction)))
	psub = subscribe!(predictive_messages |> schedule_on(m_scheduler) |> map(Message, ReactiveMP.materialize!), p_buffer)
	current_τ = GammaShapeRate(1.0, 1.0)
	current_θ = MvNormalMeanPrecision(zeros(order), 0.01 * diageye(order))
    setmarginal!(τ, current_τ)
	setmarginal!(θ, current_θ)
	ProgressMeter.@showprogress for i in 1:n-1
		for _ in 1:niter
			update!(prior_τα, shape(current_τ))
			update!(prior_τβ, rate(current_τ))
			update!(prior_mθ, mean(current_θ))
			update!(prior_vθ, precision(current_θ))

        	update!(x, x_inputs[i])
            update!(u, u_inputs[i])
            update!(z, z_inputs[i])
            update!(r, r_inputs[i])
            update!(y, outputs[i])

        	update!(x_future, x_inputs[i+1])
            update!(u_future, u_inputs[i+1])
            update!(z_future, z_inputs[i+1])
            update!(r_future, r_inputs[i+1])

			update!(y_future, missing)
			release!(f_scheduler)
	    end
		release!(m_scheduler)
		release!(fe)
		current_τ = last(τ_buffer)
		current_θ = last(θ_buffer)
	end
    return τ_buffer, θ_buffer, p_buffer, fe
end

@rule NARMAX(:θ, Marginalisation) (q_y::Missing, q_x::PointMass{Vector{Float64}}, q_u::PointMass{Vector{Float64}}, q_z::PointMass{Vector{Float64}}, q_r::PointMass{Vector{Float64}}, q_τ::GammaShapeRate{Float64}, ) = begin 
    return missing
end
@rule NARMAX(:τ, Marginalisation) (q_y::Missing, q_θ::MvNormalWeightedMeanPrecision{Float64, Vector{Float64}, Matrix{Float64}}, q_x::PointMass{Vector{Float64}}, q_u::PointMass{Vector{Float64}}, q_z::PointMass{Vector{Float64}}, q_r::PointMass{Vector{Float64}}, ) = begin 
    return missing
end
# @rule typeof(dot)(:in2, Marginalisation) (m_out::Missing, m_in1::PointMass) = begin
# 	return missing
# end
# @marginalrule typeof(dot)(:in1_in2) (m_out::Missing, m_in1::ReactiveMP.PointMass{Vector{Float64}}, m_in2::ReactiveMP.MvNormalWeightedMeanPrecision{Float64, Vector{Float64}, Matrix{Float64}}, meta::ReactiveMP.TinyCorrection) = begin
# 	return 0.0
# end


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

τ_buffer, θ_buffer, p_buffer, fe = infer_predict_narmax(inputs, outputs, 98, 25, ϕ);

local p = p_buffer[1:end-1]
plot(mean.(p), ribbon = std.(p))
plot!(outputs[2:end])
mean.(θ_buffer) |> last
plot(getvalues(fe))