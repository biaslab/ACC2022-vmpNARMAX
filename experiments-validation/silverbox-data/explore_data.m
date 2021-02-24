close all;
clearvars;
clc;

%%

load('SNLS80mV.mat')

V1=V1-mean(V1); % Remove offset errors on the input measurements (these are visible in the zero sections of the input)
                % The input is designed to have zero mean
V2=V2-mean(V2); % Approximately remove the offset errors on the output measurements. 
                % This is an approximation because the silverbox can create itself also a small DC level 
 
uSchroeder=V1(10585:10585+1023);  % select the Schroeder section of the experiment
ySchroeder=V2(10585:10585+1023);
% One period is 1024 points. Only the odd frequencies bins (f0,3f0,5f0,...) 
% are excited. f0 = fs/N, N=1024.

%% Visualization

% Range for zoom visualization
zoom = 1:40100;

% Visualize input signal
fg1 = figure(1);
scatter(1:length(V1), V1, 5, "Marker", '.')
set(fg1, 'Color', 'w', 'Position', [300 200 800 400])
title("input");
saveas(fg1, "V1.png")

% Visualize zoom of input signal
fg11 = figure(11);
plot(zoom, V1(zoom))
set(fg11, 'Color', 'w', 'Position', [300 200 800 400])
title("input");
saveas(fg11, "V1_zoom.png")

% Visualize output signal
fg2 = figure(2);
scatter(1:length(V2), V2, 5, "Marker", '.')
set(fg2, 'Color', 'w', 'Position', [300 200 800 400])
title("output");
saveas(fg2, "V2.png")

% Visualize zoom of output signal
fg21 = figure(21);
plot(zoom, V2(zoom))
set(fg21, 'Color', 'w', 'Position', [300 200 800 400])
title("output");
saveas(fg21, "V2_zoom.png")
