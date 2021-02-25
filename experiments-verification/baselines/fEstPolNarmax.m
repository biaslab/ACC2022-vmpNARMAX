function [model e] = fEstPolNarmax(data,options)

u = data.u;
y = data.y;
N = length(u);

nd = options.nd; % maximal polynomial degree
nb = options.nb; % maximal input delay
na = options.na; % maximal output delay
ne = options.ne; % maximal noise model delay
nk = nb+na+ne+1;

try ntran = options.ntran; catch; ntran = max([nb,na,ne]); end % transient length to take into account
try options.noiseCrossTerms; catch; options.noiseCrossTerms= false; end % don't allow cross terms for noise model by default
try options.crossTerms; catch; options.crossTerms= false; end % don't allow cross terms for in general by default
try options.dc; catch; options.dc= false; end % don't allow dc term for in general by default

nIter = 10;

%% generate polynomial combinations

% input and output terms
comb = [0:nd];
for ii=2:nb+na+1
    comb = [repmat(comb,1,nd+1); kron([0:nd],ones(1,size(comb,2)))];
    
    % remove combinations which have degree higher than nd
    ndComb = sum(comb);
    comb = comb(:,ndComb<=nd);
end
% noise terms
if options.noiseCrossTerms
    for ii=nb+na+1:nk
        comb = [repmat(comb,1,nd+1); kron([0:nd],ones(1,size(comb,2)))];

        % remove combinations which have degree higher than nd
        ndComb = sum(comb);
        comb = comb(:,ndComb<=nd);
    end
else
    for ii=nb+na+2:nk
%         noisecomb = [zeros(ii-1,1); 1]; % only linear terms
        noisecomb = [zeros(ii-1,nd); 1:nd];
        comb = [[comb; zeros(1,size(comb,2))] noisecomb];
    end
end
if ~options.crossTerms
    comb(:,sum(comb)>max(comb)) = [];
end
if ~options.dc
    comb(:,1) = [];
end

nComb = size(comb,2);

%% generate regressor matrix - input & output

KLin = zeros(N,nk);
for ii=0:nb
    KLin(:,ii+1) = [zeros(ii,1); u(1:end-ii)];
end
for ii=1:na
    KLin(:,nb+1+ii) = [zeros(ii,1); y(1:end-ii)];
end

%% innovation / estimation loop
e = zeros(size(u));
for iter = 1:nIter
    % complete regressor matrix
    for ii=1:ne
        KLin(:,nb+na+1+ii) = [zeros(ii,1); e(1:end-ii)];
    end

    K = ones(N,nComb);
    for ii=1:nComb
        for jj=1:nk
            K(:,ii) = K(:,ii).*KLin(:,jj).^comb(jj,ii);
        end
    end
   
    
    %% perform prediction error NARX estimate with transient removal
    
    theta = K(ntran+1:end,:)\y(ntran+1:end);

    %% compute innovation sequence
    eOld = e;
    e = y-K*theta; 
    e(1:ntran) = 0; % leave transient part out of it

%     figure; hold on;
%     plot(eOld(ntran+1:end),'b')
%     plot(e(ntran+1:end),'r')
%     plot(e(ntran+1:end)-eOld(ntran+1:end))
end

%% save model
model.nb = nb;
model.na = na;
model.ne = ne;
model.comb = comb;
model.theta = theta;