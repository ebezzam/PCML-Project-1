% comparing models to create predictions for regression data
% written by Eric Bezzam

clear all; clc; close all;
addpath('../data')

% Load data
load('Barcelona_regression.mat');

% split data
setSeed(1);
proportion = 0.8;
[~, yTr, ~, yTe] = split(y_train,X_train,proportion);

%% mean model
pred = mean(yTr)*ones(length(yTe),1);
% compute error
err_mean = (yTe - pred).^2;
errTr_mean = computeCostRMSE( yTr, mean(yTr)*ones(length(yTr),1), 1 );
errTe_mean = computeCostRMSE( yTe, pred, 1 );
fprintf('Mean Model: Train RMSE: %0.2f Test RMSE: %0.2f\n', errTr_mean, errTe_mean);

%% reorder features so that first continuous and then categorical (dummy encoded)
% extract categorical variables
catIdx = [7 18 21 22 41 46 52 54 60 65 72];
X_cat = X_train(:,catIdx);
X_d = createDummy(X_cat);
% extract continuous variables
X_cont = X_train;
X_cont(:,catIdx) = [];
nCont = size(X_cont, 2);
% put together
X = [X_cont, X_d];
% split
[XTr, yTr, XTe, yTe] = split(y_train,X,proportion);

%% linear regression - least squares
% normalize continuous features training
[ XTr_n, uX, stdX ] = normalizeFeatures( XTr(:,1:nCont) );
XTr_n = [XTr_n XTr(:,(nCont+1):end)];
% normalize test data according to stats of training data
N_te = size(XTe, 1);
XTe_n = (XTe(:,1:nCont) - repmat(uX,N_te,1))./repmat(stdX,N_te,1);
XTe_n = [XTe_n XTe(:,(nCont+1):end)];
% add bias term
tXTr = addBiasTerm(XTr_n);
tXTe = addBiasTerm(XTe_n);
% compute model
[ beta ] = leastSquares( yTr, tXTr );
% compute error
err_ls = (yTe - tXTe*beta).^2;
errTr_ls = computeCostRMSE( yTr, tXTr, beta );
errTe_ls = computeCostRMSE( yTe, tXTe, beta );
fprintf('Least Squares: Train RMSE: %0.2f Test RMSE: %0.2f\n', errTr_ls, errTe_ls);

%% gradient descent to avoid badly scaled matrix
% optimize step size using k-folds
vals = logspace(-3,1,20);
K = 10; % 10-fold CV, comment to avoid running every time
% [ alphaStar, errTrain_gd, errCV_gd ] = optimizeAlpha( yTr, XTr, vals, nCont, K );
alphaStar = 0.25;
% normalize continuous features training
N_tr = size(XTr, 1);
[ XTr_n, uX, stdX ] = normalizeFeatures( XTr(:,1:nCont) );
XTr_n = [XTr_n XTr(:,(nCont+1):end)];
% normalize test data according to stats of training data
N_te = size(XTe, 1);
XTe_n = (XTe(:,1:nCont) - repmat(uX,N_te,1))./repmat(stdX,N_te,1);
XTe_n = [XTe_n XTe(:,(nCont+1):end)];
% add bias term
tXTr = [ones(N_tr,1) XTr_n];
tXTe = [ones(N_te,1) XTe_n];
% compute error
[ beta ] = leastSquaresGD( yTr, tXTr, alphaStar );
err_gd = (yTe - tXTe*beta).^2;
errTr_gd = computeCostRMSE( yTr, tXTr, beta );
errTe_gd = computeCostRMSE( yTe, tXTe, beta );
fprintf('Gradient Descent: Train RMSE: %0.2f Test RMSE: %0.2f\n', errTr_gd, errTe_gd);


%% linear regression - ridge regression (to avoid badly scaled matrix and penalize betas)
% optimize lambda using k-folds
vals = logspace(-3,3,500);
K = 10; % 10-fold CV
[ lambdaStar, errTr, errTe ] = optimizeLambda( yTr, XTr, vals, nCont, K );
% normalize continuous features training
N_tr = size(XTr, 1);
[ XTr_n, uX, stdX ] = normalizeFeatures( XTr(:,1:nCont) );
XTr_n = [XTr_n XTr(:,(nCont+1):end)];
% normalize test data according to stats of training data
N_te = size(XTe, 1);
XTe_n = (XTe(:,1:nCont) - repmat(uX,N_te,1))./repmat(stdX,N_te,1);
XTe_n = [XTe_n XTe(:,(nCont+1):end)];
% add bias term
tXTr = [ones(N_tr,1) XTr_n];
tXTe = [ones(N_te,1) XTe_n];
% compute error
[ beta ] = ridgeRegression( yTr, tXTr, lambdaStar );
err_rr = (yTe - tXTe*beta).^2;
errTr_rr = computeCostRMSE( yTr, tXTr, beta );
errTe_rr = computeCostRMSE( yTe, tXTe, beta );
% show results
fprintf('Ideal lambda for ridge regression: %0.2f\n', lambdaStar);
fprintf('Ridge Regression: Train RMSE: %0.2f Test RMSE: %0.2f\n', errTr_rr, errTe_rr);

figure(2)
semilogx(vals, errTr, vals, errTe)
hx = xlabel('\lambda, regularization parameter');
hy = ylabel('RMSE');
xlim([10^-3 10^3])
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.01 .01],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
grid minor;
legend('training', 'cross-validation', 'Location', 'NorthWest')



%% linear regression - ridge regression (multinomial)
% optimize lambda using k-folds
regVals = logspace(-3,3,100);
degreeVals = 1:7;
K = 10; % 10-fold CV
[ degreeStar, lambdaStar, errTr, errTe ] = optimizeDegreeLambda( yTr, XTr, degreeVals, regVals, nCont, K );
% normalize and crete poly matric
N_tr = size(XTr, 1);
[ XTr_n, uX, stdX ] = normalizeFeatures( XTr(:,1:nCont) );
XTr_poly = [myPoly(XTr_n,degreeStar), XTr(:,(nCont+1):end)];
% normalize test data according to stats of training data and make poly
% matrix
N_te = size(XTe, 1);
XTe_n = (XTe(:,1:nCont) - repmat(uX,N_te,1))./repmat(stdX,N_te,1);
XTe_poly = [myPoly(XTe_n,degreeStar) XTe(:,(nCont+1):end)];
% add bias term
tXTr = [ones(N_tr,1) XTr_poly];
tXTe = [ones(N_te,1) XTe_poly];
% compute error
[ beta ] = ridgeRegression( yTr, tXTr, lambdaStar );
err_poly = (yTe - tXTe*beta).^2;
errTr_poly = computeCostRMSE( yTr, tXTr, beta );
errTe_poly = computeCostRMSE( yTe, tXTe, beta );
% show results
fprintf('Ideal lambda for polynomial ridge regression: %0.2f\n', lambdaStar);
fprintf('Ideal degree: %0.2f\n', degreeStar);
fprintf('Polynomial Ridge Regression: Train RMSE: %0.2f Test RMSE: %0.2f\n', errTr_poly, errTe_poly);

figure(3)
plot(degreeVals, errTr, degreeVals, errTe)
hx = xlabel('M, polynomial degree');
hy = ylabel('RMSE');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.01 .01],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
grid minor;
legend('training', 'cross-validation', 'Location', 'NorthEast')

%% multiple ridge regression models
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
% train each model - optimize lambda and degree using k-folds
regVals = logspace(-3,3,100);
degreeVals = 1:7;
K = 4; % 4-fold CV
[ degreeStar1, lambdaStar1, errTr1, errTe1 ] = optimizeDegreeLambda( y1, X1, degreeVals, regVals, nCont, K );
[ degreeStar2, lambdaStar2, errTr2, errTe2 ] = optimizeDegreeLambda( y2, X2, degreeVals, regVals, nCont, K );
[ degreeStar3, lambdaStar3, errTr3, errTe3 ] = optimizeDegreeLambda( y3, X3, degreeVals, regVals, nCont, K );

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
predErr1 = y1 - tX1*beta1; 
predErr2 = y2 - tX2*beta2; 
predErr3 = y3 - tX3*beta3; 
pred = [predErr1; predErr2; predErr3];
N_tr = length(pred);
errTr = sqrt(pred'*pred/(N_tr));

% compute test error
% assign test data to groups
[ g1, g2, g3 ] = assignGroups( mu, XTe(:,3), XTe(:,19) );
% form each group
X1 = XTe(g1,:);
y1 = yTe(g1,:);
X2 = XTe(g2,:);
y2 = yTe(g2,:);
X3 = XTe(g3,:);
y3 = yTe(g3,:);
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
% test error
predErr1 = y1 - tX1*beta1; 
predErr2 = y2 - tX2*beta2; 
predErr3 = y3 - tX3*beta3; 
pred = [predErr1; predErr2; predErr3];
N_te = length(pred);
errTe = sqrt(pred'*pred/(N_te));

fprintf('Multiple model approach: Train RMSE: %0.2f Test RMSE: %0.2f\n', errTr, errTe);


%% comparing mean and gradient descent
figure(101); 
boxplot([err_mean(:), err_gd(:)]); 
grid on; 
title('Error Distribution of Mean (1) and Linear Regression (2) Models');
ylabel('Squared Error - (y_{true}-y_{pred})^2')
xlabel('Model')

%% comparing poly and ridge regression
figure(102); 
boxplot([err_rr(:), err_poly(:)]); 
hx = xlabel('Model');
hy = ylabel('Squared Error');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.01 .01],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;

%% compare poly and multiple
figure(103); 
err_mult = pred.^2;
boxplot([err_poly(:), err_mult(:)]); 
ylim([-0.25e06 2e06])
hx = xlabel('Model');
hy = ylabel('Squared Error');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.01 .01],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
