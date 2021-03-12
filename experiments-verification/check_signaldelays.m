close all;
clear all;
clc;

% addpath(genpath("baselines"));

%% Set options

options.nb = 1; % input delays
options.na = 1; % output delays
options.ne = 1; % innovation delays
options.nd = 3; % degree polynomial nonlinearity

options.stdu = 1;
options.stde = .03;

%% Generate input

options.N = 2^16;
options.P = 1;
options.M = 1;
options.fMin = 0;
options.fMax = 100;
options.fs = 1000;
options.type =  'odd';

[uTrain, ~] = fMultiSinGen(options);
[uTest, ~] = fMultiSinGen(options); % other random realization
uTest = options.stdu*uTest; % make test signal slightly smaller than training
uTrain = options.stdu*uTrain;

dataSysTrain.u = uTrain;
dataSysTrain.e = options.stde*randn(size(uTrain));

dataSysTest.u = uTest;
dataSysTest.e = options.stde*randn(size(uTest));

dataSysTest0.u = uTest;
dataSysTest0.e = zeros(size(uTest));

%% Generate output

dataTemp.u = randn(100,1);
dataTemp.y = randn(100,1);
[modelTemp,~] = fEstPolNarmax(dataTemp,options);

sysComb = modelTemp.comb;
% % remove all crossterms
% sysComb(:,sum(sysComb)>max(sysComb)) = [];
% sysComb(:,1) = []; % remove dc component
nComb = size(sysComb,2);

[b,a] = butter(options.nb,0.1);% set linear components equal to butterworth filter

sysTheta = zeros(nComb,1);
sysTheta(2:options.nd:options.nd*(options.nb+1+options.na)) = 0.01*(rand(options.nb+1+options.na,1)-0.5); % even terms
sysTheta(3:options.nd:options.nd*(options.nb+1+options.na)) = 0.01*(rand(options.nb+1+options.na,1)-0.5); % odd terms
sysTheta(1:options.nd:(options.nb+1)*options.nd)=b;
sysTheta((options.nb+1)*options.nd+1:options.nd:(options.nb+1)*options.nd+options.na*options.nd)=-a(2:end);
sysTheta((options.nb+1+options.na)*options.nd+1:options.nd:(options.nb+1+options.na)*options.nd+options.ne*options.nd)=0.1;
sysTheta(end-options.nd+2:end)=100*(rand(options.nd-1,1)-0.5); % nl terms noise

ToySystem.nb = options.nb;
ToySystem.na = options.na;
ToySystem.ne = options.na;
ToySystem.comb = sysComb;
ToySystem.theta = sysTheta;


yTrain_delay1 = fSimPolNarmax(dataSysTrain,ToySystem);

yTest_delay1 = fSimPolNarmax(dataSysTest,ToySystem);

yTest0_delay1 = fSimPolNarmax(dataSysTest0,ToySystem);

%%

zoom_t = 1:200;

f1 = figure();
clf();
% subplot(2,1,1)
plot(zoom_t, uTrain(zoom_t), 'LineWidth', 3)
hold on
% subplot(2,1,2)
plot(zoom_t, yTrain_delay1(zoom_t), 'LineWidth', 2)
legend(["output", "input"])
xlabel('time [k]');
ylabel('Amplitude');
title("Delays = " + string(options.na));
set(gcf, 'Color', 'w', 'Position', [200 200 600 300]);
saveas(f1, "genNARMAX_delays"+string(options.na)+".png");


%%

options.nb = 3; % input delays
options.na = 3; % output delays
options.ne = 3; % innovation delays

dataTemp.u = randn(100,1);
dataTemp.y = randn(100,1);
[modelTemp,~] = fEstPolNarmax(dataTemp,options);

sysComb = modelTemp.comb;
% % remove all crossterms
% sysComb(:,sum(sysComb)>max(sysComb)) = [];
% sysComb(:,1) = []; % remove dc component
nComb = size(sysComb,2);

[b,a] = butter(options.nb,0.1);% set linear components equal to butterworth filter

sysTheta = zeros(nComb,1);
sysTheta(2:options.nd:options.nd*(options.nb+1+options.na)) = 0.01*(rand(options.nb+1+options.na,1)-0.5); % even terms
sysTheta(3:options.nd:options.nd*(options.nb+1+options.na)) = 0.01*(rand(options.nb+1+options.na,1)-0.5); % odd terms
sysTheta(1:options.nd:(options.nb+1)*options.nd)=b;
sysTheta((options.nb+1)*options.nd+1:options.nd:(options.nb+1)*options.nd+options.na*options.nd)=-a(2:end);
sysTheta((options.nb+1+options.na)*options.nd+1:options.nd:(options.nb+1+options.na)*options.nd+options.ne*options.nd)=0.1;
sysTheta(end-options.nd+2:end)=100*(rand(options.nd-1,1)-0.5); % nl terms noise

ToySystem.nb = options.nb;
ToySystem.na = options.na;
ToySystem.ne = options.na;
ToySystem.comb = sysComb;
ToySystem.theta = sysTheta;


yTrain_delay3 = fSimPolNarmax(dataSysTrain,ToySystem);

yTest_delay3 = fSimPolNarmax(dataSysTest,ToySystem);

yTest0_delay3 = fSimPolNarmax(dataSysTest0,ToySystem);

%%

zoom_t = 1:300

f2 = figure();
clf();
% subplot(2,1,1)
plot(zoom_t, uTrain(zoom_t), 'LineWidth', 2)
hold on
% subplot(2,1,2)
plot(zoom_t, yTrain_delay3(zoom_t), 'LineWidth', 2)
legend(["input", "output"])
xlabel('time [k]');
ylabel('Amplitude');
title("Delays = 3");
set(gcf, 'Color', 'w', 'Position', [200 200 600 300]);
saveas(f2, "genNARMAX_delays3.png");
yl = get(gca, 'yLim');

f3 = figure();
clf();
% subplot(2,1,1)
plot(zoom_t, uTrain(zoom_t), 'LineWidth', 2)
hold on
% subplot(2,1,2)
plot(zoom_t, yTrain_delay1(zoom_t), 'LineWidth', 2)
legend(["input", "output"])
xlabel('time [k]');
ylabel('Amplitude');
set(gca, 'yLim', yl);
title("Delays = 1");
set(gcf, 'Color', 'w', 'Position', [200 200 600 300]);
saveas(f3, "genNARMAX_delays1.png");
