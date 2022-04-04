close all;
clear all;

%% Specify options

stdu = 0.1;
stde = .005;

options.na = 3; % # output delays
options.nb = 3; % # input delays
options.ne = 3; % # innovation delays
options.nd = 1; % # degree polynomial nonlinearity
options.N = 2^16;
options.P = 1;
options.M = 1;
options.fMin = 0;
options.fMax = 100;
options.fs = 1000;
options.type =  'odd';

%% Generate

[uTrain, ~] = fMultiSinGen(options);

plot(uTrain)