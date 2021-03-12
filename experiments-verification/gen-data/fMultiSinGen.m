function [u lines] = fMultiSinGen(options)
% 
% generates a zero-mean random phase multisine with std = 1
% INPUT
% options.N: number of points per period
% options.P: number of periods
% options.M: number of realizations
% options.fMin: minimum excited frequency
% options.fMax: maximum escited frequency
% options.fs: sample frequency
% options.type: 'full', 'odd', 'oddrandom'
%
% OPTIONAL
% options.nGroup: in case of oddrandom, 1 out of nGroup odd lines is
%                 discarded. Default = 3
% options.std: std of the generated signals. Default = 1
%
% OUTPUT
% u: NPxM record of the generated signals
% lines: excited frequency lines -> 1 = dc, 2 = fs/N
% 
% copyright:
% Maarten Schoukens
% Vrije Universiteit Brussel, Brussels Belgium
% 10/05/2017
%
% This work is licensed under a 
% Creative Commons Attribution-NonCommercial 4.0 International License
% (CC BY-NC 4.0)
% https://creativecommons.org/licenses/by-nc/4.0/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% settings
try nGroup = options.nGroup; catch, nGroup = 3; end
try uStd = options.std; catch, uStd = 1; end

N = options.N;
P = options.P;
M = options.M;
fs = options.fs;

f0 = fs/N;

%% lines selection - select which frequencies to excite
linesMin = ceil(options.fMin/f0)+1;
linesMax = floor(options.fMax/f0)+1;
lines = linesMin:linesMax;

% remove dc
if lines(1) == 1;
    lines = lines(2:end);
end

switch options.type
    case 'full'
        % do nothing
    case 'odd'
        % remove even lines - odd indices
        if mod(lines(1),2) % lines(1) is odd
           lines = lines(2:2:end); 
        else
           lines = lines(1:2:end); 
        end
    case 'oddrandom'
        % remove even lines - odd indices
        if mod(lines(1),2) % lines(1) is odd
           lines = lines(2:2:end); 
        else
           lines = lines(1:2:end); 
        end
        % remove 1 out of nGroup lines
        nLines = length(lines);
        nRemove = floor(nLines/nGroup);
        removeInd = randi(nGroup,[1 nRemove]);
        removeInd = removeInd + nGroup*[0:nRemove-1];
        lines(removeInd) = [];
end
nLines = length(lines);

%% multisine generation - frequency domain implementation
U = zeros(N,M);
U(lines,:) = exp(2i*pi*rand(nLines,M)); % excite the selected frequencies
u = real(ifft(U)); % go to time domain
u = uStd*u./std(u(:,1)); % rescale to obtain desired rms std

u = repmat(u,[P,1]); % generate P periods