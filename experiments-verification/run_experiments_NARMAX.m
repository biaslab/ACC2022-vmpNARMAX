close all;
clear all;
clc;

% addpath(genpath("baselines"));

%%

% Series of train sizes
trn_sizes = [100, 500, 1000, 5000];
num_trnsizes = length(trn_sizes);

% Define transient and test indices
transient = 1000;
ix_tst = 1:1000 + transient;

options.na = 3; % # output delays
options.nb = 3; % # input delays
options.ne = 3; % # innovation delays
options.nd = 3; % # degree polynomial nonlinearity

M_m = options.na + 1 + options.nb + options.ne;

options.N = 2^16;
options.P = 1;
options.M = 1;

options.fMin = 0;
options.fMax = 100;
options.fs = 1000;
options.type = 'odd';

options.stdu = 0.5;
options.stde = .01;

% Number of repetitions
num_repeats = 100;

% Preallocate result arrays
RMS_prd = zeros(num_repeats, num_trnsizes);
RMS_sim = zeros(num_repeats, num_trnsizes);

r = 1;
while r <= num_repeats
    waitbar(r/num_repeats)
    
    % Generate signal
    [yTrain, yTest, uTrain, uTest] = gen_signal(options);
    
    if max(yTrain) < 1e3
    
        % Write signal to file
        save("data/NARMAXsignal_r" + string(r) + ".mat", "yTrain", "yTest", "uTrain", "uTest", "options")
        
        for n = 1:length(trn_sizes)
            
            % Establish length of training signal
            ix_trn = 1:trn_sizes(n) + transient;
        
            % Slice data
            dataTrain.u = uTrain(ix_trn);
            dataTrain.y = yTrain(ix_trn);
            dataTest.u = uTest(ix_tst);
            dataTest.y = yTest(ix_tst);

            % KLS estimator
            [modelNarmaxIter,eNarmaxIter] = fEstPolNarmax(dataTrain,options);

            % 1-step ahead prediction
            [yPredIterTest,ePredIterTest] = fPredPolNarmax(dataTest,modelNarmaxIter);

            % Simulation
            ySimIterTest = fSimPolNarmax(dataTest,modelNarmaxIter);

            % Compute RMS
            RMS_prd(r,n) = rms(dataTest.y - yPredIterTest);
            RMS_sim(r,n) = rms(dataTest.y - ySimIterTest);
            
            % Write results to file
            save("results/results-NARMAX_FEM_M"+num2str(M_m)+"_degree3_S"+string(length(ix_trn))+".mat", "RMS_prd", "RMS_sim")
            
        end
        
        % Increment repeat
        r = r + 1;
    end    
end

disp("RMS");
[nanmean(RMS_prd,1); nanmean(RMS_sim,1)]

disp("Proportion instable");
[sum(isnan(RMS_prd)); sum(isnan(RMS_sim))]
