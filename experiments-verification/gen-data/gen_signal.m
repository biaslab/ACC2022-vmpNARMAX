function [yTrain, yTest, uTrain, uTest, ToySystem] = gen_signal(options)

% Unpack options
na = options.na;
nb = options.nb;
ne = options.ne;
nd = options.nd;

stde = options.stde;
stdu = options.stdu;

%% Start NARX system

% Define system
dataTemp.u = randn(100,1);
dataTemp.y = randn(100,1);
[modelTemp,~] = fEstPolNarmax(dataTemp,options);

sysComb = modelTemp.comb;
% % remove all crossterms
% sysComb(:,sum(sysComb)>max(sysComb)) = [];
% sysComb(:,1) = []; % remove dc component
nComb = size(sysComb,2);

[b,a] = butter(nb,0.1);% set linear components equal to butterworth filter

sysTheta = zeros(nComb,1);
sysTheta(2:nd:nd*(nb+1+na)) = 0.01*(rand(nb+1+na,1)-0.5); % even terms
sysTheta(3:nd:nd*(nb+1+na)) = 0.01*(rand(nb+1+na,1)-0.5); % odd terms
sysTheta(1:nd:(nb+1)*nd) = b;
sysTheta((nb+1)*nd+1:nd:(nb+1)*nd+na*nd) = -a(2:end);
sysTheta((nb+1+na)*nd+1:nd:(nb+1+na)*nd+ne*nd) = 0.1;
sysTheta(end-nd+2:end) = 0.1*(rand(nd-1,1)-0.5); % nl terms noise

ToySystem.nb = nb;
ToySystem.na = na;
ToySystem.ne = na;
ToySystem.nd = nd;
ToySystem.stde = stde;
ToySystem.comb = sysComb;
ToySystem.theta = sysTheta;

%% Generate input

[uTrain, ~] = fMultiSinGen(options);
[uTest, ~] = fMultiSinGen(options); % other random realization
uTest = stdu*uTest; % make test signal slightly smaller than training
uTrain = stdu*uTrain;

dataSysTrain.u = uTrain;
dataSysTrain.e = stde*randn(size(uTrain));
dataSysTest.u = uTest;
dataSysTest.e = stde*randn(size(uTest));

%% Generate output

yTrain = fSimPolNarmax(dataSysTrain,ToySystem);
yTest = fSimPolNarmax(dataSysTest,ToySystem);

if options.normalize
   
    max_value = max(abs([yTrain; yTest; uTrain; uTest]));
    yTrain = yTrain ./ max_value;
    yTest = yTest ./ max_value;
    uTrain = uTrain ./ max_value;
    uTest = uTest ./ max_value;
    
end

end