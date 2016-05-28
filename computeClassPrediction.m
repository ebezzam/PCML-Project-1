% compute predictions for classification data using PLR
% written by Eric Bezzam

clear all; clc; close all;
addpath('code')
addpath('data')

% Load data
load('Barcelona_classification.mat');
y_train(y_train==-1)=0;

XTr = X_train;
yTr = y_train;
XTe = X_test;

%% training - basic feature processing
% extract categorical variables
catIdx = [8 24 32 35 37];
X_cat = X_train(:,catIdx);
X_d = createDummy(X_cat);
% extract continuous variables
X_cont = X_train;
X_cont(:,catIdx) = [];
nCont = size(X_cont, 2);
% put together
XTr = [X_cont];

%% test - basic feature processing
% extract categorical variables
catIdx = [8 24 32 35 37];
X_cat = X_train(:,catIdx);
X_d = createDummy(X_cat);
% extract continuous variables
X_cont = X_test;
X_cont(:,catIdx) = [];
nCont = size(X_cont, 2);
% put together
XTe = [X_cont];

%% calculate model using all of training data
alpha = 0.4;
lambda = 0.001;

% normalize continuous features training
[ XTr_n, uX, stdX ] = normalizeFeatures( XTr(:,1:nCont) );
XTr_n = [XTr_n XTr(:,(nCont+1):end)];

% add bias term
tXTr = addBiasTerm(XTr_n);

% compute model
[ beta ] = penLogisticRegression( yTr, tXTr, alpha, lambda );

% compute error
errTrRMSE = computeCostRMSEClass( yTr, tXTr, beta );
errTr01 = computeCostZeroOne( yTr, tXTr, beta);
errTrlog = computeCostLogLoss( yTr, tXTr, beta );


%% compute prediction probabilities

% normalize test data according to stats of training data
N_te = size(XTe, 1);
XTe_n = (XTe(:,1:nCont) - repmat(uX,N_te,1))./repmat(stdX,N_te,1);
XTe_n = [XTe_n XTe(:,(nCont+1):end)];

% add bias term
tXTe = addBiasTerm(XTe_n);

% prediction probabilities 
pred = sigmoid(tXTe*beta);

csvwrite('predictions/predictions_classification.csv', pred);