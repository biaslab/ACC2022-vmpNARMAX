close all;
clear all;
clc;

%%

load('silverbox-data/SNLS80mV.mat')

options.fs = 610.35; % Hz
options.na = 3; % # output delays
options.nb = 3; % # input delays
options.ne = 3; % # innovation delays
options.nd = 3; % # degree polynomial nonlinearity

options.dc = true;

M = options.na + 1 + options.nb + options.ne;

% Select training set
iTrain = 4.05*1e4:131072;
input_trn = V1(iTrain)';
output_trn = V2(iTrain)';

% Select validation set
iTest = 1e3:(4.05*1e4-1); % start at 1e3 to avoid transient
input_tst = V1(iTest)';
output_tst = V2(iTest)';

% Slice data
dataTrain.u = input_trn;
dataTrain.y = output_trn;
dataTest.u = input_tst;
dataTest.y = output_tst;

% ILS estimator
[modelNarmaxIter,eNarmaxIter] = fEstPolNarmax(dataTrain,options);

% 1-step ahead prediction
[yPredIterTest,ePredIterTest] = fPredPolNarmax(dataTest,modelNarmaxIter);

% Simulation
ySimIterTest = fSimPolNarmax(dataTest,modelNarmaxIter);

% Compute RMS
RMS_prd_ILS = rms(dataTest.y - yPredIterTest);
RMS_sim_ILS = rms(dataTest.y - ySimIterTest);

save("results/silverbox-NARMAX-ILS_order" + num2str(M) + "_results.mat", "yPredIterTest", "ySimIterTest", "modelNarmaxIter", "RMS_prd_ILS", "RMS_sim_ILS")

%%

figure; hold on;
plot(dataTest.y)
plot(dataTest.y - ySimIterTest)
% plot(dataTest.y - yPredIterTest)
legend('system output','simulation error','prediction error')

disp('  1-Step Ahead Prediction')
disp(['  RMS Test Error: ' num2str(rms(dataTest.y - yPredIterTest)*1e3)])

disp('  Simulation')
disp(['  RMS Test Error: ' num2str(rms(dataTest.y - ySimIterTest)*1e3)])
