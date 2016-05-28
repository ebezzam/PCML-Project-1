% compute predictions for regression data using multi-model approach
% written by Eric Bezzam

clear all; clc; close all;
addpath('code')
addpath('data')

% Load data
load('Barcelona_regression.mat');
XTr = X_train;
yTr = y_train;
XTe = X_test;

%% training data - reorder features so that first continuous and then categorical (dummy encoded) 
% extract categorical variables
catIdx = [7 18 21 22 41 46 52 54 60 65 72];
X_cat = XTr(:,catIdx);
X_d = createDummy(X_cat);
% extract continuous variables
X_cont = X_train;
X_cont(:,catIdx) = [];
nCont = size(X_cont, 2);
% put together
XTr = [X_cont, X_d];

%% test data - reorder features so that first continuous and then categorical (dummy encoded) 
% extract categorical variables
catIdx = [7 18 21 22 41 46 52 54 60 65 72];
X_cat = XTe(:,catIdx);
X_d = createDummy(X_cat);
% extract continuous variables
X_cont = X_test;
X_cont(:,catIdx) = [];
% put together
XTe = [X_cont, X_d];

%% calculate model using all of training data

% use K-means to calculate centers of 3 clusters
K = 3;
mu_0 = [-1 -1; -1 5; 3 5];
[ mu, g1, g2, g3 ] = computeCentroids(XTr(:,3), XTr(:,19), K, mu_0);

% form each group
X1 = XTr(g1,:);
y1 = yTr(g1,:);
X2 = XTr(g2,:);
y2 = yTr(g2,:);
X3 = XTr(g3,:);
y3 = yTr(g3,:);

% model parameter calculated from compareRegModels.m
degreeStar1 = 2;
degreeStar2 = 1;
degreeStar3 = 3;
lambdaStar1 = 13.3194;
lambdaStar2 = 327.4549;
lambdaStar3 = 1000;

% create polynomial input vector for each group
N1 = size(X1, 1);   % first group
[ X_n, uX1, stdX1 ] = normalizeFeatures( X1(:,1:nCont) );
X_poly = [myPoly(X_n,degreeStar1), X1(:,(nCont+1):end)];
tX1 = [ones(N1,1) X_poly];
N2 = size(X2, 1);   % second group
[ X_n, uX2, stdX2 ] = normalizeFeatures( X2(:,1:nCont) );
X_poly = [myPoly(X_n,degreeStar2), X2(:,(nCont+1):end)];
tX2 = [ones(N2,1) X_poly];
N3 = size(X3, 1);   % third group
[ X_n, uX3, stdX3 ] = normalizeFeatures( X3(:,1:nCont) );
X_poly = [myPoly(X_n,degreeStar3), X3(:,(nCont+1):end)];
tX3 = [ones(N3,1) X_poly];

% create model for each group
[ beta1 ] = ridgeRegression( y1, tX1, lambdaStar1 );
[ beta2 ] = ridgeRegression( y2, tX2, lambdaStar2 );
[ beta3 ] = ridgeRegression( y3, tX3, lambdaStar3 );

% compute training error
pred = ones(length(yTr),1);
pred(g1) = tX1*beta1; 
pred(g2) = tX2*beta2;
pred(g3) = tX3*beta3; 
errTr =  computeCostRMSE( yTr, pred, 1 );


%% compute predictions

% assign test data to groups
[ g1, g2, g3 ] = assignGroups( mu, XTe(:,3), XTe(:,19) );

% form each group
X1 = XTe(g1,:);
X2 = XTe(g2,:);
X3 = XTe(g3,:);

% create polynomial input vector for each group
N1 = size(X1, 1); % first group
X_n = (X1(:,1:nCont) - repmat(uX1,N1,1))./repmat(stdX1,N1,1);
X_poly = [myPoly(X_n,degreeStar1), X1(:,(nCont+1):end)];
tX1 = [ones(N1,1) X_poly];
N2 = size(X2, 1);
X_n = (X2(:,1:nCont) - repmat(uX2,N2,1))./repmat(stdX2,N2,1);
X_poly = [myPoly(X_n,degreeStar2), X2(:,(nCont+1):end)];
tX2 = [ones(N2,1) X_poly];
N3 = size(X3, 1);
X_n = (X3(:,1:nCont) - repmat(uX3,N3,1))./repmat(stdX3,N3,1);
X_poly = [myPoly(X_n,degreeStar3), X3(:,(nCont+1):end)];
tX3 = [ones(N3,1) X_poly];

% predictions
pred = ones(size(XTe,1),1);
pred(g1) = tX1*beta1; 
pred(g2) = tX2*beta2;
pred(g3) = tX3*beta3;

csvwrite('predictions/predictions_regression.csv', pred);
