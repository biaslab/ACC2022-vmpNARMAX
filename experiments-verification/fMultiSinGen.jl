using SpecialFunctions
using Statistics
using FFTW

function fMultiSinGen(N::Integer,
                      P::Integer,
                      M::Integer;
                      fMin::Float64=0.99,
                      fMax::Float64=1.00,
                      fs::Float64=1.0,
                      type_signal::String="full",
                      nGroup::Integer=3,
                      uStd::Float64=1.)
    """
    generates a zero-mean random phase multisine with std = 1
    INPUT
    options.N: number of points per period
    options.P: number of periods
    options.M: number of realizations
    options.fMin: minimum excited frequency
    options.fMax: maximum escited frequency
    options.fs: sample frequency
    options.type: "full", "odd", "oddrandom"
    
    OPTIONAL
    options.nGroup: in case of oddrandom, 1 out of nGroup odd lines is
                     discarded. Default = 3
    options.std: std of the generated signals. Default = 1
    
    OUTPUT
    u: NPxM record of the generated signals
    lines: excited frequency lines -> 1 = dc, 2 = fs/N
     
    copyright:
    Maarten Schoukens
    Vrije Universiteit Brussel, Brussels Belgium
    10/05/2017
    
    translated from Matlab to Julia by
    Wouter Kouw
    TU Eindhoven, Eindhoven, Netherlands
    22/01/2021
    
    This work is licensed under a 
    Creative Commons Attribution-NonCommercial 4.0 International License
    (CC BY-NC 4.0)
    https://creativecommons.org/licenses/by-nc/4.0/
    """

    # Lines selection - select which frequencies to excite
    f0 = fs/N
    linesMin = Int64(ceil(fMin / f0) + 1)
    linesMax = Int64(floor(fMax / f0) + 1)
    lines = linesMin:linesMax

    # Remove DC component
    if lines[1] == 1; lines = lines[2:end]; end

    if type_signal == "full"
        # do nothing
    elseif type_signal == "odd"
        
        # remove even lines - odd indices
        if Bool(mod(lines[1],2)) # lines(1) is odd
            lines = lines[2:2:end]
        else
            lines = lines[1:2:end]
        end

    elseif type_signal == "oddrandom"
        
        # remove even lines - odd indices
        if Bool(mod(lines[1],2)) # lines(1) is odd
            lines = lines[2:2:end]
        else
            lines = lines[1:2:end]
        end
        
        # remove 1 out of nGroup lines
        nLines = length(lines)
        nRemove = floor(nLines / nGroup)
        removeInd = rand(1:nGroup, [1 nRemove])
        removeInd = removeInd + nGroup*[0:nRemove-1]
        lines = lines(!removeInd)
    end
    nLines = length(lines)

    # multisine generation - frequency domain implementation
    U = zeros(ComplexF64, N,M)

    # excite the selected frequencies
    U[lines,:] = exp.(2im*pi*rand(nLines,M))
    
    # go to time domain
    u = real(ifft(U))
    
    # rescale to obtain desired rms std
    u = uStd * u ./ std(u[:,1])

    # generate P periods
    u = repeat(u, outer=(P,1))

    return u, lines
end