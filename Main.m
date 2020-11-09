%-------------------------------------------------------------------------%
%  Binary Dragonfly Algorithm (BDA) source codes demo version             %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%


%---Inputs-----------------------------------------------------------------
% feat:   features
% label:  labelling
% N:      Number of dragonflies
% T:      Maximum number of iterations
%---Outputs----------------------------------------------------------------
% sFeat:  Selected features
% Sf:     Selected feature index
% Nf:     Number of selected features
% curve:  Convergence curve
%--------------------------------------------------------------------------


%% Binary Dragonfly Algorithm
clc; clear; close
% Benchmark data set 
load ionosphere.mat;
% Set 20% data as validation set
ho=0.2; 
% Hold-out method
HO=cvpartition(label,'HoldOut',ho,'Stratify',false);
% Parameter setting
N=10; T=100; 
% Perform feature selection 
[sFeat,Sf,Nf,curve]=jBDA(feat,label,N,T,HO);
% Accuracy 
Acc=jKNN(sFeat,label,HO); 
% Plot convergence curve
figure(); plot(1:T,curve); xlabel('Number of Iterations');
ylabel('Fitness Value'); title('BDA'); grid on;




