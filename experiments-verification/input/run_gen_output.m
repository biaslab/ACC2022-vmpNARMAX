close all;
clearvars;
clc;

%% Generate input

N = 4000;

options.N = N;
options.P = 10;
options.M = 1;
options.fMin = 0;
options.fMax = 100;
options.fs = 1000;
options.type =  'odd';

[u, ~] = fMultiSinGen(options);

%% Generate output

% Delay orders (M1 = output, M2 = input, M3 = errors)
orders.M1 = 2;
orders.M2 = 2;
orders.M3 = 2;
M = orders.M1 + 1 + orders.M2 + orders.M3;

% Nonlinearity
degree = 3;
N = (orders.M1 + 1 + orders.M2 + orders.M3)*degree + 1;

PP = zeros(M,1); 
for d = 1:degree
    PP = [d .*eye(M,M), PP]; 
end
phi = @(x) prod(cell2mat(arrayfun(@(k) x.^PP(:,k), 1:size(PP,2), 'UniformOutput', false)), 1)';

% Parameters
params.theta = 0.1 .*(rand(N,1) - 0.5);
params.tau = 1e5;

% Generate output
y = gen_output(u, phi, params, orders);

%% Visualize

figure(1)
plot(u, 'Color', 'r');
xlabel('time [k]');
title('input');

figure(2)
plot(y, 'Color', 'k');
xlabel('time [k]');
title('output');

figure(3)
hold on
t_zoom = 100:200;
plot(t_zoom, u(t_zoom), 'Color', 'r');
plot(t_zoom, y(t_zoom), 'Color', 'k');
xlabel('time [k]');
legend('input', 'output')

