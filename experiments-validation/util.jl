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