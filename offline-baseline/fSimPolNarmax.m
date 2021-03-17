function ySim = fSimPolNarmax(data,model)

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
