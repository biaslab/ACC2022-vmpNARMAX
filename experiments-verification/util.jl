using LinearAlgebra


function gen_combs(options)

    na = options["na"]
    nb = options["nb"]
    ne = options["ne"]
    nd = options["nd"]
    nk = nb+na+ne+1
    
    # Repeat powers
    comb = reshape(collect(0:nd), (1,nd+1))

    # Start combinations array
    combs = reshape(collect(0:nd), (1,nd+1))
    for ii = 2:nb+na+1

        # Current width
        width = size(combs,2)

        # Increment combinations array
        combs = [repeat(combs,1,nd+1); kron(comb,ones(1,width))]
        
        # remove combinations which have degree higher than nd
        ndComb = sum(combs,dims=1)
        combs = combs[:, vec(ndComb .<= nd)]
    end

    if options["noiseCrossTerms"]
        for ii = nb+na+2:nk

            # Current width
            width = size(combs,2)

            # Add noise cross terms
            combs = [repeat(combs,1,nd+1); kron(comb,ones(1,width))]
    
            # remove combinations which have degree higher than nd
            ndComb = sum(combs, dims=1)
            combs = combs[:, vec(ndComb .<= nd)]
        end
    else
        for ii = nb+na+2:nk
    #         noisecomb = [zeros(ii-1,1); 1]; % only linear terms
            noisecomb = [zeros(ii-1, nd); reshape(collect(1:nd), 1,nd)]
            combs = [[combs; zeros(1,size(combs,2))] noisecomb]
        end
    end

    if !options["crossTerms"]
        combs = combs[:, vec(sum(combs,dims=1) .> maximum(combs,dims=1))]
    end
    
    if !options["dc"]
        combs = combs[:,2:end]
    end

    return combs
end

function tmean(x::AbstractArray, dims::Int64; tr::Real=0.2)
    """`tmean(x; tr=0.2)`
    Trimmed mean of real-valued array `x`.
    Find the mean of `x`, omitting the lowest and highest `tr` fraction of the data.
    This requires `0 <= tr <= 0.5`. The amount of trimming defaults to `tr=0.2`.
    """

    nrows,ncols = size(x)

    if dims==1
        tmeans = zeros(1,ncols)
        for n = 1:nrows
            tmeans[1,n] = tmean!(copy(x[:,n]), tr=tr)
        end
    elseif dims==2
        tmeans = zeros(nrows,1)
        for n = 1:ncols
            tmeans[n,1] = tmean!(copy(x[n,:]), tr=tr)
        end
    end
    return tmeans
end

function tmean(x::AbstractArray; tr::Real=0.2)
    """`tmean(x; tr=0.2)`
    Trimmed mean of real-valued array `x`.
    Find the mean of `x`, omitting the lowest and highest `tr` fraction of the data.
    This requires `0 <= tr <= 0.5`. The amount of trimming defaults to `tr=0.2`.
    """
    if size(x,1) == 0
        return NaN
    else
        tmean!(copy(x), tr=tr)
    end
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
    if size(x,1) == 0
        return NaN
    else
        return sqrt(winvar(x,tr=tr))/((1-2tr)*sqrt(length(x)))
    end
end

function winval(x::AbstractArray; tr::Real=0.2)
    """`winval(x; tr=0.2)`
    Winsorize real-valued array `x`.
    Return a copy of `x` in which extreme values (that is, the lowest and highest
    fraction `tr` of the data) are replaced by the lowest or highest non-extreme
    value, as appropriate. The trimming fraction defaults to `tr=0.2`.
    """
    n = length(x)
    if n == 0
        return NaN
    else     
        xcopy   = sort(x)
        ibot    = floor(Int64, tr*n)+1
        itop    = n-ibot+1
        xbot, xtop = xcopy[ibot], xcopy[itop]
        return  [x[i]<=xbot ? xbot : (x[i]>=xtop ? xtop : x[i]) for i=1:n]
    end
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