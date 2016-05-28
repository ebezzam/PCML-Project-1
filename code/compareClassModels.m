% comparing models to create predictions for regression data
% written by Eric Bezzam

clear all; clc; close all;
addpath('../data')

% Load data
load('Barcelona_classification.mat');
y_train(y_train==-1)=0;

%% basic feature processing
setSeed(1);
proportion = 0.8;
% extract categorical variables
catIdx = [8 24 32 35 37];
X_cat = X_train(:,catIdx);
% X_d = createDummy(X_cat);
% extract continuous variables
X_cont = X_train;
X_cont(:,catIdx) = [];
% out = [7 24];
% X_cont(:,out) = [];
nCont = size(X_cont, 2);
% put together
X = [X_cont];
% split
[XTr, yTr, XTe, yTe] = split(y_train,X,proportion);

%% simple model
pred_tr = median(yTr)*ones(length(yTr),1); % predict most common output
N_tr = length(yTr);
pred_te = median(yTr)*ones(length(yTe),1);
N_te = length(yTe);
% compute error
err_const = (yTe - pred_te).^2;
errTrRMSE_const = sqrt((yTr - pred_tr)'*(yTr - pred_tr)/N_tr);
errTeRMSE_const = sqrt((yTe - pred_te)'*(yTe - pred_te)/N_te);
errTr01_const = computeCostZeroOne( yTr, pred_tr, 1 );
errTe01_const = computeCostZeroOne( yTe, pred_te, 1 );
fprintf('Simple Model: Train RMSE: %0.4f Test RMSE: %0.4f\n', errTrRMSE_const, errTeRMSE_const);
fprintf('Simple Model: Train 0-1loss: %0.4f Test 0-1loss: %0.4f\n', errTr01_const, errTe01_const);


%% logistic regression
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
alpha = 0.4;
[ beta ] = logisticRegression( yTr, tXTr, alpha );
% compute error
err_lr = (yTe - sigmoid(tXTe*beta)).^2;
errTrRMSE_lr = computeCostRMSEClass( yTr, tXTr, beta );
errTeRMSE_lr = computeCostRMSEClass( yTe, tXTe, beta );
errTr01_lr = computeCostZeroOne( yTr, tXTr, beta );
errTe01_lr = computeCostZeroOne( yTe, tXTe, beta );
errTrlog_lr = computeCostLogLoss( yTr, tXTr, beta );
errTelog_lr = computeCostLogLoss( yTe, tXTe, beta );
fprintf('\nLR Model: Train RMSE: %0.4f Test RMSE: %0.4f\n', errTrRMSE_lr, errTeRMSE_lr);
fprintf('LR Model: Train 0-1loss: %0.4f Test 0-1loss: %0.4f\n', errTr01_lr, errTe01_lr);
fprintf('LR Model: Train logLoss: %0.4f Test logLoss: %0.4f\n', errTrlog_lr, errTelog_lr);


%% penalized logistic regression
% optimize regularization parameters
alpha = 0.4;
regVals = linspace(0.001, 0.5, 10);
K = 5;
[ lambdaStar, errTr, errTe ] = optimizeLambdaClass( yTr, XTr, alpha, regVals, nCont, K );
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
[ beta ] = penLogisticRegression( yTr, tXTr, alpha, lambdaStar );
% compute error
err_plr = (yTe - sigmoid(tXTe*beta)).^2;
errTrRMSE_lr = computeCostRMSEClass( yTr, tXTr, beta );
errTeRMSE_lr = computeCostRMSEClass( yTe, tXTe, beta );
errTr01_lr = computeCostZeroOne( yTr, tXTr, beta);
errTe01_lr = computeCostZeroOne( yTe, tXTe, beta );
errTrlog_lr = computeCostLogLoss( yTr, tXTr, beta );
errTelog_lr = computeCostLogLoss( yTe, tXTe, beta );
fprintf('\nPLR Model: Train RMSE: %0.4f Test RMSE: %0.4f\n', errTrRMSE_lr, errTeRMSE_lr);
fprintf('PLR Model: Train 0-1loss: %0.4f Test 0-1loss: %0.4f\n', errTr01_lr, errTe01_lr);
fprintf('PLR Model: Train logLoss: %0.4f Test logLoss: %0.4f\n', errTrlog_lr, errTelog_lr);
fprintf('Ideal lambda for penalized logistic regression %0.4f\n', lambdaStar);

% visualize cross-validation
figure(1)
plot(regVals, errTr, regVals, errTe)
hx = xlabel('\lambda, regularization parameter');
hy = ylabel('0-1 loss');
xlim([10^-3 10^3])
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.01 .01],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
xlim([min(regVals) max(regVals)])
grid minor;
legend('training', 'cross-validation', 'Location', 'NorthWest')

%% multinomial penalized logistic regression
% optimize lambda using k-folds
alpha = 0.1;
regVals = linspace(0.001, 0.1, 5);
degreeVals = 2;
K = 5;
[ degreeStar, lambdaStar, errTr, errTe ] = optimizeClassDegreeLambda( yTr, XTr, alpha, degreeVals, regVals, nCont, K );
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
% compute model
[ beta ] = penLogisticRegression( yTr, tXTr, alpha, lambdaStar );
% compute error
err_mult = (yTe - sigmoid(tXTe*beta)).^2;
errTrRMSE_lr = computeCostRMSEClass( yTr, tXTr, beta );
errTeRMSE_lr = computeCostRMSEClass( yTe, tXTe, beta );
errTr01_lr = computeCostZeroOne( yTr, tXTr, beta);
errTe01_lr = computeCostZeroOne( yTe, tXTe, beta );
errTrlog_lr = computeCostLogLoss( yTr, tXTr, beta );
errTelog_lr = computeCostLogLoss( yTe, tXTe, beta );
fprintf('\nMultinomial PLR Model: Train RMSE: %0.4f Test RMSE: %0.4f\n', errTrRMSE_lr, errTeRMSE_lr);
fprintf('Multinomial PLR Model: Train 0-1loss: %0.4f Test 0-1loss: %0.4f\n', errTr01_lr, errTe01_lr);
fprintf('Multinomial PLR Model: Train logLoss: %0.4f Test logLoss: %0.4f\n', errTrlog_lr, errTelog_lr);
fprintf('Ideal lambda for multi PLR %0.4f\n', lambdaStar);
fprintf('Ideal degree for multiPLR %0.4f\n', degreeStar);

figure(2)
plot(degreeVals, errTr, degreeVals, errTe)
hx = xlabel('M, polynomial degree');
hy = ylabel('0-1 loss');
set(gca,'fontsize',20,'fontname','Helvetica','box','off','tickdir','out','ticklength',[.01 .01],'xcolor',0.5*[1 1 1],'ycolor',0.5*[1 1 1]);
set([hx; hy],'fontsize',18,'fontname','avantgarde','color',[.3 .3 .3]);
grid on;
grid minor;
legend('training', 'cross-validation', 'Location', 'East')

%% compare models
figure(3)
boxplot([err_lr(:), err_plr(:), err_mult(:)])