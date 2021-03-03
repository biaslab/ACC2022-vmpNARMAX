close all;
clear all;
clc;

% addpath(genpath("baselines"));

%%

transient = 1:10;
iTrain = 1:1000 + transient(end);
iTest = 1:1000 + iTrain(end);

options.na = 3; % # output delays
options.nb = 3; % # input delays
options.ne = 3; % # innovation delays
options.nd = 3; % # degree polynomial nonlinearity

options.N = 2^16;
options.P = 1;
options.M = 1;

options.fMin = 0;
options.fMax = 100;
options.fs = 1000;
options.type = 'odd';

options.stdu = .1;
options.stde = .01;

% Number of repetitions
num_repeats = 10;

% Preallocate result arrays
RMS_prd = zeros(num_repeats,1);
RMS_sim = zeros(num_repeats,1);

r = 1;
while r <= num_repeats
    waitbar(r/num_repeats)
    
    % Generate signal
    [yTrain, yTest, uTrain, uTest] = gen_signal(options);
    
    if max(yTrain) < 1e3
    
        % Write signal to file
        save("data/NARMAXsignal_r" + string(r) + ".mat", "yTrain", "yTest", "uTrain", "uTest", "options")
        
        % Slice data
        dataTrain.u = uTrain(iTrain);
        dataTrain.y = yTrain(iTrain);
        dataTest.u = uTest(iTest);
        dataTest.y = yTest(iTest);
        
        % KLS estimator
        [modelNarmaxIter,eNarmaxIter] = fEstPolNarmax(dataTrain,options);
        
        % 1-step ahead prediction
        [yPredIterTest,ePredIterTest] = fPredPolNarmax(dataTest,modelNarmaxIter);
        
        % Simulation
        ySimIterTest = fSimPolNarmax(dataTest,modelNarmaxIter);

        % Compute RMS
        RMS_prd(r) = rms(dataTest.y - yPredIterTest);
        RMS_sim(r) = rms(dataTest.y - ySimIterTest);
        
        % Increment repeat
        r = r + 1;
    end    
end

[RMS_prd RMS_sim]

