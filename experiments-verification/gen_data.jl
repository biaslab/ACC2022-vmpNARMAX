using Random

include("fMultiSinGen.jl")


function gensignalARX(θ_true, τ_true; M1=2, M2=2, fMin=0.8, fMax=1.0, fs=1.0, T=100)

    # Orders
    M = M1+1+M2

    # Parameters
    θ_true = θ_scale .*(rand(M,) .- 0.5)

    # Input frequency and amplitude
    input, _ = fMultiSinGen(T, 1, 1, fMin=fMin, fMax=fMax, fs=fs, type_signal="odd")

    # Observation array
    output = zeros(T,)
    errors = zeros(T,)

    for k = 1:T
        
        # Input
        input[k] = A[k]*cos(ω*k)
        
        # Generate noise
        errors[k] = sqrt(inv(τ_true))*randn(1)[1]
    
        # Output
        if k < M
            output[k] = input[k] + errors[k]
        else
            # Update history vectors
            x_kmin1 = output[k-1:-1:k-M1]
            z_kmin1 = input[k-1:-1:k-M2]
            
            # Compute output
            output[k] = θ_true'*[x_kmin1; input[k]; z_kmin1] + errors[k]
        end
    end

    return input, output
end


function gensignalNARX(ϕ, θ_scale, τ_true; M1=2, M2=2, degree=1, fMin=0.8, fMax=1.0, fs=1.0, T=100)

    # Orders
    M = (M1+1+M2)*degree + 1

    # Parameters
    θ_true = θ_scale .*(rand(M,) .- 0.5)
    θ_true[end] = 0.

    # Input frequency and amplitude
    input, _ = fMultiSinGen(T, 1, 1, fMin=fMin, fMax=fMax, fs=fs, type_signal="odd")

    # Observation array
    output = zeros(T,)
    errors = zeros(T,)

    for k = 1:T
        
        # Input
        input[k] = A[k]*cos(ω*k)
        
        # Generate noise
        errors[k] = sqrt(inv(τ_true))*randn(1)[1]
    
        # Output
        if k < M
            output[k] = input[k] + errors[k]
        else
            # Update history vectors
            x_kmin1 = output[k-1:-1:k-M1]
            z_kmin1 = input[k-1:-1:k-M2]
            
            # Compute output
            output[k] = θ_true'*ϕ([x_kmin1; input[k]; z_kmin1]) + errors[k]
        end
    end

    return input, output
end

function gensignalNARMAX(ϕ, θ_scale, τ_true; M1=2, M2=2, M3=2, degree=1, fMin=0.8, fMax=1.0, fs=1.0, uStd=1., T=100, num_periods=1, points_period=1)

    # Orders
    M = (M1+1+M2+M3)*degree + 1

    # Parameters
    θ_true = θ_scale .*(rand(M,) .- 0.5)
    θ_true[end] = 0.0

    # Input frequency and amplitude
    input, _ = fMultiSinGen(points_period, num_periods, 1, fMin=fMin, fMax=fMax, fs=fs, type_signal="full", uStd=uStd)

    # Observation array
    output = zeros(T,)
    errors = zeros(T,)

    for k = 1:T
        
        # Generate noise
        errors[k] = sqrt(inv(τ_true))*randn(1)[1]
    
        # Output
        if k < (maximum([M1,M2,M3])+1)
            output[k] = input[k] + errors[k]
        else
            # Update history vectors
            x_kmin1 = output[k-1:-1:k-M1]
            z_kmin1 = input[k-1:-1:k-M2]
            r_kmin1 = errors[k-1:-1:k-M3]
            
            # Compute output
            output[k] = θ_true'*ϕ([x_kmin1; input[k]; z_kmin1; r_kmin1]) + errors[k]
        end
    end

    return input, output
end


function split_data(T, split_index; start_index=10)

    # Check inputs
    if split_index > T
        error("Split index "*string(split_index)*"must be lower than end time T="*string(T))
    end

    # Select training set
    ix_trn = Array{Int64,1}(start_index:split_index)

    # Select validation set
    ix_val = Array{Int64,1}(split_index+1:T)

    return ix_trn, ix_val
end