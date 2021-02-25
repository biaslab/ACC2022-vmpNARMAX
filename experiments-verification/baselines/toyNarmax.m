% Silverbox Polynomial NARMAX Model Estimation

close all
clear all
clc

%% set variables
fs = 1000; % Hz
iTrain = 4.05*1e4:131072;
iTest = 1e3:(4.05*1e4-1); % start at 1e3 to avoid transient
iTestNoExtra = 1e3:(3*1e4); % start at 1e3 to avoid transient
timeTrain = [0:(length(iTrain)-1)]/fs;

nb = 1; % # input delays
na = nb; % # output delays
ne = na; % # innovation delays
nd = 3; % # degree polynomial nonlinearity

N = 2^16;
P = 2;

stdu = 1;
stde = .03;

%% Define system
dataTemp.u = randn(100,1);
dataTemp.y = randn(100,1);
options.nb = nb;
options.na = na;
options.ne = ne;
options.nd = nd;
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
sysTheta(1:nd:(nb+1)*nd)=b;
sysTheta((nb+1)*nd+1:nd:(nb+1)*nd+na*nd)=-a(2:end);
sysTheta((nb+1+na)*nd+1:nd:(nb+1+na)*nd+ne*nd)=0.1;
sysTheta(end-nd+2:end)=100*(rand(nd-1,1)-0.5); % nl terms noise

ToySystem.nb = nb;
ToySystem.na = na;
ToySystem.ne = na;
ToySystem.comb = sysComb;
ToySystem.theta = sysTheta;

%% Generate output
options.N = N;
options.P = P;
options.M = 1;
options.fMin = 0;
options.fMax = 100;
options.fs = fs;
options.type =  'odd';

[uTrain, ~] = fMultiSinGen(options);
[uTest, ~] = fMultiSinGen(options); % other random realization
uTest = stdu*uTest; % make test signal slightly smaller than training
uTrain = stdu*uTrain;

dataSysTrain.u = uTrain;
dataSysTrain.e = stde*randn(size(uTrain));
yTrain = fSimPolNarmax(dataSysTrain,ToySystem);

dataSysTest.u = uTest;
dataSysTest.e = stde*randn(size(uTest));
yTest = fSimPolNarmax(dataSysTest,ToySystem);

dataSysTest0.u = uTest;
dataSysTest0.e = zeros(size(uTest));
yTest0 = fSimPolNarmax(dataSysTest0,ToySystem);

figure; plot(yTest); hold on; plot(yTest-yTest0); shg

%% estimate iterative scheme
dataTrain.u = uTrain(:);
dataTrain.y = yTrain(:);
options.nb = nb;
options.na = na;
options.ne = ne;
options.nd = nd;
[modelNarmaxIter,eNarmaxIter] = fEstPolNarmax(dataTrain,options);

%% validate
[yPredIterTrain,ePredIterTrain] = fPredPolNarmax(dataTrain,modelNarmaxIter);
ySimIterTrain = fSimPolNarmax(dataTrain,modelNarmaxIter);

[yPredIterTrain0,ePredIterTrain0] = fPredPolNarmax(dataTrain,ToySystem);
ySimIterTrain0 = fSimPolNarmax(dataTrain,ToySystem);

dataTest.u = uTest(:);
dataTest.y = yTest(:);

[yPredIterTest,ePredIterTest] = fPredPolNarmax(dataTest,modelNarmaxIter);
ySimIterTest = fSimPolNarmax(dataTest,modelNarmaxIter);

[yPredIterTest0,ePredIterTest0] = fPredPolNarmax(dataTest,ToySystem);
ySimIterTest0 = fSimPolNarmax(dataTest,ToySystem);

%% plot

figure; hold on;
plot(yTest)
plot(yTest(:)-ySimIterTest(:))
plot(yTest(:)-yPredIterTest(:))
legend('system output','simulation error','prediction error')

disp('  1-Step Ahead Prediction')
disp(['  model RMS Training Error: ' num2str(rms(yTrain-yPredIterTrain))])
disp(['  model RMS Test Error: ' num2str(rms(yTest-yPredIterTest))])

disp(['  system RMS Training Error: ' num2str(rms(yTrain-yPredIterTrain0))])
disp(['  system RMS Test Error: ' num2str(rms(yTest-yPredIterTest0))])

disp('  Simulation')
disp(['  model RMS Training Error: ' num2str(rms(yTrain-ySimIterTrain))])
disp(['  model RMS Test Error: ' num2str(rms(yTest-ySimIterTest))])

disp(['  system RMS Training Error: ' num2str(rms(yTrain-ySimIterTrain0))])
disp(['  system RMS Test Error: ' num2str(rms(yTest-ySimIterTest0))])