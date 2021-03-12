close all;
clear all;
clc;

% addpath(genpath("baselines"));

%%

options.na = 1; % # output delays
options.nb = 1; % # input delays
options.ne = 1; % # innovation delays
options.nd = 3; % # degree polynomial nonlinearity

options.N = 2^16;
options.P = 1;
options.M = 1;

options.fMin = 0;
options.fMax = 100;
options.fs = 1000;
options.type = 'odd';

options.stdu = 1.0;
options.stde = .03;

options.normalize = false;

% Generate signal
[yTrain, yTest, uTrain, uTest, theta] = gen_signal(options);

%%

zoom_t = 1:200;

f1 = figure();
clf();
hold on
plot(zoom_t, uTrain(zoom_t), 'LineWidth', 2)
plot(zoom_t, yTrain(zoom_t), 'LineWidth', 2)
legend(["input", "output"])
xlabel('time [k]');
ylabel('Amplitude');
title("Delays = " + string(options.na));
set(gcf, 'Color', 'w', 'Position', [200 200 600 300]);
saveas(f1, "genNARMAX_delays"+string(options.na)+".png");

f2 = figure();
clf();
bar(theta)


