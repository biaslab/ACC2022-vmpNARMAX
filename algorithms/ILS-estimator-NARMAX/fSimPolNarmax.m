function ySim = fSimPolNarmax(data,model)
    % 
% Estimates coefficients for a polynomial NARMAX model
% INPUT
% data.u: input signal
% model.na: number of delays for outputs
% model.nb: number of delays for inputs
% model.ne: number of delays for errors   
% model.nd: maximal polynomial degree
% model.comb: array with combinations of polynomial orders
%
% OUTPUT
% ySim: simulated outputs
% 
% copyright:
% Maarten Schoukens
% Vrije Universiteit Brussel, Brussels Belgium
% 18/03/2021
%
% This work is licensed under a 
% Creative Commons Attribution-NonCommercial 4.0 International License
% (CC BY-NC 4.0)
% https://creativecommons.org/licenses/by-nc/4.0/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% if no e sequence provided, e = 0;
try e = data.e; catch; e = zeros(size(data.u)); end

u = data.u;
N = length(u);

nb = model.nb;
na = model.na;
ne = model.ne;
nk = nb+na+ne+1;

comb = model.comb;
nComb = size(comb,2);

%% simulation
nc = max([nb,na,ne]);
eSim = [zeros(nc,1); e(:)]; % zeropadding for unknown initial conditions
uSim = [zeros(nc,1); u(:)]; % zeropadding for unknown initial conditions
ySim = zeros(size(uSim));
for ii=nc+1:N+nc
    % construct regressor vector
    KLin = [uSim(ii:-1:ii-nb); ySim(ii-1:-1:ii-na); eSim(ii-1:-1:ii-ne);].'; % row vector

    K = ones(1,nComb);
    for kk=1:nComb
        for jj=1:nk
            K(1,kk) = K(1,kk).*(KLin(1,jj).^comb(jj,kk));
        end
    end
    ySim(ii) = K*model.theta + eSim(ii);
end
ySim = ySim(nc+1:end); %remove zero padding part    
