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
% kfold:  Number of K-fold cross-validation
% k:      Number of k in KNN
%---Outputs----------------------------------------------------------------
% sFeat:  Selected features
% Sf:     Selected feature index
% Nf:     Number of selected features
% curve:  Convergence curve
%--------------------------------------------------------------------------


%% Binary Dragonfly Algorithm
clc; clear; close
% Set parameters
kfold=10; k=5; N=10; T=100;
O.k=k; O.kfold=kfold; O.N=N; O.T=T; 
% Load data
load ionosphere.mat; 
% Divide data into train & validate using cross-validation
CV=cvpartition(label,'KFold',kfold,'Stratify',true);
O.Model=CV; 
% Perform feature selection 
[sFeat,Sf,Nf,curve]=jBDA(feat,label,O);
% Accuracy 
Acc=jKNN(sFeat,label,CV,O); 
% Plot convergence curve
figure(); plot(1:T,curve); xlabel('Number of Iterations');
ylabel('Fitness Value'); title('BDA'); grid on;




