function [yPred,ePred] = fPredPolNarmax(data,model)
% 
% Estimates coefficients for a polynomial NARMAX model
% INPUT
% data.u: input signal
% data.y: output signal
% model.na: number of delays for outputs
% model.nb: number of delays for inputs
% model.ne: number of delays for errors   
% model.nd: maximal polynomial degree
% model.comb: array with combinations of polynomial orders
%
% OUTPUT
% yPred: predicted outputs
% ePred: prediction errors
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

u = data.u;
y = data.y;
N = length(u);

nb = model.nb;
na = model.na;
ne = model.ne;
nk = nb+na+ne+1;

comb = model.comb;
nComb = size(comb,2);

%% simulation
nc = max([nb,na,ne]);
u = [zeros(nc,1); u(:)]; % zeropadding for unknown initial conditions
y = [zeros(nc,1); y(:)]; % zeropadding for unknown initial conditions
e = zeros(size(u));
yPred = zeros(size(u));
for ii=nc+1:N+nc
    % construct regressor vector
    KLin = [u(ii:-1:ii-nb); y(ii-1:-1:ii-na); e(ii-1:-1:ii-ne)]'; % row vector

    K = ones(1,nComb);
    for kk=1:nComb
        for jj=1:nk
            K(1,kk) = K(1,kk).*(KLin(1,jj).^comb(jj,kk));
        end
    end
    
    yPred(ii) = K*model.theta;
    
    e(ii) = (y(ii) - yPred(ii));

end
yPred = yPred(nc+1:end); %remove zero padding part    
ePred = e(nc+1:end); %remove zero padding part 

