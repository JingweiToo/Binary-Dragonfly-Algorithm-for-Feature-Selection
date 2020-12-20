%-------------------------------------------------------------------%
%  Binary Dragonfly Algorithm (BDA) demo version                    %
%-------------------------------------------------------------------%


%---Inputs-----------------------------------------------------------
% feat     : feature vector (instances x features)
% label    : label vector (instances x 1)
% N        : Number of dragonflies
% max_Iter : Maximum number of iterations

%---Outputs-----------------------------------------------------------
% sFeat    : Selected features
% Sf       : Selected feature index
% Nf       : Number of selected features
% curve    : Convergence curve
%---------------------------------------------------------------------


%% Binary Dragonfly Algorithm
clc; clear; close
% Benchmark data set 
load ionosphere.mat; 

% Set 20% data as validation set
ho = 0.2; 
% Hold-out method
HO = cvpartition(label,'HoldOut',ho,'Stratify',false);

% Parameter setting
N        = 10; 
max_Iter = 100; 
% Perform feature selection 
[sFeat,Sf,Nf,curve] = jBDA(feat,label,N,max_Iter,HO);

% Accuracy 
Acc = jKNN(sFeat,label,HO);

% Plot convergence curve
plot(1:max_Iter,curve);
xlabel('Number of Iterations');
ylabel('Fitness Value'); 
title('BDA'); grid on;


