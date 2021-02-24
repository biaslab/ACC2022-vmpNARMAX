using Plots
pyplot()


function plot_forecast(output, predictions, ix_trn, ix_val; tt=1, posterior=false)

    T = length(output)

    # Limits
    ylims = [minimum(output[ix_trn[1]:end]), maximum(output[ix_trn[1]:end])]

    # Plot training output
    plot(ix_trn[1:tt:end], output[ix_trn[1:tt:end]], color="blue", ylims=ylims, label="training data", size=(800,300))

    # Plot validation output
    scatter!(ix_val[1:tt:end], output[ix_val[1:tt:end]], color="black", ylims=ylims, label="validation data")

    # Plot fit and forecast
    if posterior
        sdev = sqrt.(predictions[2][ix_trn[1]:tt:T])
        plot!(ix_trn[1]:tt:T, predictions[1][ix_trn[1]:tt:T], ribbon=[sdev, sdev], color="purple", label="predictions")
    else
        plot!(ix_trn[1]:tt:T, predictions[ix_trn[1]:tt:T], color="purple", label="predictions")
    end
end


function plot_errors(output, predictions, ix_val)

    # Compute prediction errors
    pred_errors = (predictions[ix_val] - output[ix_val]).^2

    # Compute root mean square
    RMS = sqrt(mean(pred_errors))

    # Plot errors of time
    scatter(pred_errors, label="RMS = "*string(RMS), ylabel="squared error", yscale=:log10, size=(800,300))
end

function tmean(x::AbstractArray; tr::Real=0.2)
    """`tmean(x; tr=0.2)`
    Trimmed mean of real-valued array `x`.
    Find the mean of `x`, omitting the lowest and highest `tr` fraction of the data.
    This requires `0 <= tr <= 0.5`. The amount of trimming defaults to `tr=0.2`.
    """
    tmean!(copy(x), tr=tr)
end

function tmean!(x::AbstractArray; tr::Real=0.2)
    """`tmean!(x; tr=0.2)`
    Trimmed mean of real-valued array `x`, which sorts the vector `x` in place.
    Find the mean of `x`, omitting the lowest and highest `tr` fraction of the data.
    This requires `0 <= tr <= 0.5`. The trimming fraction defaults to `tr=0.2`.
    """
    if tr < 0 || tr > 0.5
        error("tr cannot be smaller than 0 or larger than 0.5")
    elseif tr == 0
        return mean(x)
    elseif tr == .5
        return median!(x)
    else
        n   = length(x)
        lo  = floor(Int64, n*tr)+1
        hi  = n+1-lo
        return mean(sort!(x)[lo:hi])
    end
end

function trimse(x::AbstractArray; tr::Real=0.2)
    """`trimse(x; tr=0.2)`
    Estimated standard error of the mean for Winsorized real-valued array `x`.
    See `winval` for what Winsorizing (clipping) signifies.
    """
    return sqrt(winvar(x,tr=tr))/((1-2tr)*sqrt(length(x)))
end

function winval(x::AbstractArray; tr::Real=0.2)
    """`winval(x; tr=0.2)`
    Winsorize real-valued array `x`.
    Return a copy of `x` in which extreme values (that is, the lowest and highest
    fraction `tr` of the data) are replaced by the lowest or highest non-extreme
    value, as appropriate. The trimming fraction defaults to `tr=0.2`.
    """
    n = length(x)
    xcopy   = sort(x)
    ibot    = floor(Int64, tr*n)+1
    itop    = n-ibot+1
    xbot, xtop = xcopy[ibot], xcopy[itop]
    return  [x[i]<=xbot ? xbot : (x[i]>=xtop ? xtop : x[i]) for i=1:n]
end

function winmean(x::AbstractArray; tr=0.2)
    """`winmean(x; tr=0.2)`
    Winsorized mean of real-valued array `x`.
    See `winval` for what Winsorizing (clipping) signifies.
    """
    return mean(winval(x, tr=tr))
end

function winvar(x::AbstractArray; tr=0.2)
    """`winvar(x; tr=0.2)`
    Winsorized variance of real-valued array `x`.
    See `winval` for what Winsorizing (clipping) signifies.
    """
    return var(winval(x, tr=tr))
end

function winstd(x::AbstractArray; tr=0.2)
    """`winstd(x; tr=0.2)`
    Winsorized standard deviation of real-valued array `x`.
    See `winval` for what Winsorizing (clipping) signifies.
    """
    return std(winval(x, tr=tr))
end

function wincov(x::AbstractArray, y::AbstractArray; tr::Real=0.2)
    """`wincov(x, y; tr=0.2)`
    Compute the Winsorized covariance between `x` and `y`.
    See `winval` for what Winsorizing (clipping) signifies.
    """
    xvec = winval(x, tr=tr)
    yvec = winval(y, tr=tr)
    return cov(xvec, yvec)
end