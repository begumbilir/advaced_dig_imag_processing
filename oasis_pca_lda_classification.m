% OASIS: dataset with Mild Cognitive Impairment patients and healthy controls
%
% Data are available in small and large format:
%   - subs_05: small, subsampled to 50%
%   - subs_10: large, original size of 100%
% Effect of Age and Total Intracranial Volume (eTIV) regressed out:
%   - dataset          : original data
%       vol contains 4D data
%   - residual_dataset : effects regressed out
%       resid_vol contains 4D data
clear all
close all

%datadir='E:\data\oasis\' 

load oasis_residual_dataset_subs_05_20150309T105732_19924
%load oasis_dataset_subs_05_20150309T105732_19924


%% prepare data
sz = size(resid_vol); % size of data
lab = stats.CDR*2;    % get group label: 0 is controls, 1 is MCI patients

age = stats.Age;


resid_vol_sampled = resid_vol(1:3:end, 1:3:end, 1:3:end, :);


sz_sampled = size(resid_vol_sampled);
tmp_sampled = reshape(resid_vol_sampled,prod(sz_sampled(1:3)),[]);

% or run this line for residual

m_p = sum(lab > 0); % number of patients
m_c = sum(lab == 0); % number of controls
m = m_p + m_c; %total number of participants
n = length(tmp_sampled); 


tmp_p = tmp_sampled(:,lab > 0); % vectorized voxels for MRI scans of Alzhemier patients
tmp_c = tmp_sampled(:, lab == 0); % vectorized voxels for MRI scans of healthy (control) group


% Split the r balanced by keeping most of the initial class distribution (2k for control, k for patients) 


function x_star = PCA_LDA(r,n,m_p, m_c, m, lab_selected, tmp_sampled_selected)
%[P_mtrx, sing_vals] = svd(tmp_sampled_selected, 'econ');



%[P_mtrx, sing_vals] = svd(tmp_sampled_selected, 'econ');
[U, S, ~] = svd(tmp_sampled_selected, 'econ');
P_mtrx_r = U(:, 1:r);

% Reverse columns of U manually (no fliplr)
P_mtrx = P_mtrx_r(:, end:-1:1);




Scattered_P = zeros(n,n);
tmp_prime = zeros(r, m);

% x_mean = zeros(n,1);
% for i = 1:m
%     x_mean = x_mean + tmp_sampled_selected(:,i);
% end 
% x_mean = x_mean./r;

for i = 1:m
    x_i = tmp_sampled_selected(:,i);
    x_mean = mean(tmp_sampled_selected, 2);  % mean along rows
    x_i_centered = x_i - x_mean;
    %disp(size(x_mean))
    %x_i_centered = x_i - mean(x_i) * ones(n,1);
    % Create Scattered matrix
    Scattered_P = Scattered_P + x_i_centered * transpose(x_i_centered);
    % Projecting x_i onto P matrix
    x_i_prime = transpose(P_mtrx) * x_i_centered;
    tmp_prime(:,i) = x_i_prime;
end

x_p_prime_mean = zeros(r ,1);
x_c_prime_mean = zeros(r ,1);
for i = 1:m
    x_i_prime = tmp_prime(:,i);
    if lab_selected(i) == 1 %if patient
        x_p_prime_mean = x_p_prime_mean + x_i_prime;
    else %if control
        x_c_prime_mean = x_c_prime_mean + x_i_prime;
    end     
end

x_p_prime_mean = x_p_prime_mean ./ m_p;
x_c_prime_mean = x_c_prime_mean ./ m_c;

x_prime_mean = (m_p .* x_p_prime_mean + m_c .* x_c_prime_mean) ./ m; 

sw_p = zeros(r,r);
sw_c = zeros(r,r);
for i = 1:m
   x_i_prime = tmp_prime(:,i);
   if lab_selected(i) == 1 %if patient
       x_temp = x_i_prime - x_p_prime_mean;
       sw_p = sw_p + x_temp * transpose(x_temp);
   else %if control
       x_temp = x_i_prime - x_c_prime_mean;
       sw_c = sw_c + x_temp * transpose(x_temp);
   end 
end 

Scattered_w = (sw_p + sw_c)./ m;

x_p_norm = x_p_prime_mean - x_prime_mean;
x_c_norm = x_c_prime_mean - x_prime_mean;

Scattered_b = ((m_p.*(x_p_norm )*transpose(x_p_norm )) +...
             (m_c.*(x_c_norm)*transpose(x_c_norm)))./m;

q_mtrx = inv(Scattered_w)*Scattered_b;
[W, D] = eig(q_mtrx);
[eigvals, order] = sort(diag(D), 'descend');  % sort eigenvalues largest → smallest
q_leading = W(:, order(1)); % leading eigenvector (r×1)
a = P_mtrx*q_leading; 
%fprintf('a to be plotted');
%disp(a)
x_star = zeros(r,1);
for i = 1:m 
    x_i = tmp_sampled_selected(:,i);
    x_mean = mean(tmp_sampled_selected, 2);  % mean along rows
    
    x_star_i = transpose(real(a))*(x_i-x_mean);
    x_star(i,1) = x_star_i;
    %disp (x_star_i)
end

end 


%classification with svm and 5 fold cross validation 
rVals      = 10:10:180;          % 10, 20, …, 180
numR       = numel(rVals);
trainAccSVM   = zeros(numR,1);      % training accuracies
cvErrSVM      = zeros(numR,1); 



r = 150;  % number of prinicpal components (chosen as number of total MRI scans)


for i = 1:numR
    
    r =  rVals(i);

    x_star = PCA_LDA(r,n,m_p, m_c, m, lab, tmp_sampled);
    %disp(x_star)
    SVMModel = fitcsvm(x_star, lab,'KernelFunction', 'linear','Standardize',true,'ClassNames',[0 1]);
    CVSVMModel = crossval(SVMModel, 'KFold', 5);
    cvLoss = kfoldLoss(CVSVMModel);
    %fprintf('5-fold CV classification error: %.2f%%\n', cvLoss*100);
    [labelPred, scorePred] = predict(SVMModel, x_star);
    trainAcctemp = mean(labelPred == lab);

    cvErrSVM(i)     = cvLoss;     % fraction mis-classified
    trainAccSVM(i)  = trainAcctemp;   % fraction correct

end 
%%
figure
yyaxis left
plot(10:10:180,trainAccSVM*100,'-o','LineWidth',1.5, ...
     'DisplayName','Training accuracy')
ylabel('Accuracy (%)')

yyaxis right
plot(10:10:180,cvErrSVM*100,'-s','LineWidth',1.5, ...
     'DisplayName','5-fold CV error')
ylabel('CV error (%)')

xlabel('r (number of features / PCs)')
title('SVM performance vs. r')
legend('Location','best')
grid on

%%
rVals    = 10:15:180;          % 10, 20, …, 180
numR     = numel(rVals);
nIter    = 10;                 % number of repetitions per r

% pre‐allocate storage: rows=r’s, cols=iterations
trainAccMat = zeros(numR, nIter);
cvErrMat    = zeros(numR, nIter);

for i = 1:numR
    r = rVals(i);
    for it = 1:nIter
        % 1) get 1‐D features via PCA+LDA on *all* data
        x_star = PCA_LDA(r, n, m_p, m_c, m, lab, tmp_sampled);

        % 2) fit & cv‐evaluate SVM
        SVMModel   = fitcsvm(x_star, lab,       ...
                             'KernelFunction','linear', ...
                             'Standardize', true,      ...
                             'ClassNames',  [0 1]);
        CVSVMModel = crossval(SVMModel, 'KFold', 5);
        cvLoss     = kfoldLoss(CVSVMModel);    % fraction mis‐classified

        % 3) training accuracy on same data
        yhatTrain  = predict(SVMModel, x_star);
        trainAcc   = mean(yhatTrain == lab);

        % store
        trainAccMat(i, it) = trainAcc;
        cvErrMat   (i, it) = cvLoss;
    end
end

% compute means and standard deviations
trainAccMean = mean(trainAccMat,  2);
trainAccStd  = std (trainAccMat,  0, 2);
cvErrMean    = mean(cvErrMat,     2);
cvErrStd     = std (cvErrMat,     0, 2);

%% Plot mean curves with error‐bars (optional)
figure; 
yyaxis left
errorbar(rVals, trainAccMean*100, trainAccStd*100, '-o', 'LineWidth',1.5);
ylabel('Training accuracy (%)');
yyaxis right
errorbar(rVals, cvErrMean*100, cvErrStd*100, '-s', 'LineWidth',1.5);
ylabel('CV error (%)');
xlabel('r (number of PCs)');
title('SVM performance vs. r (mean \pm 1 std over 10 runs)');
legend({'Train Acc','CV Err'}, 'Location','best');
grid on;

%% Plot *just* the standard deviations vs. r
figure;
plot(rVals, trainAccStd*100, '-o', 'LineWidth',1.5, 'DisplayName','Train Acc STD');
hold on;
plot(rVals, cvErrStd*100,    '-s', 'LineWidth',1.5, 'DisplayName','CV Err STD');
hold off;
xlabel('r (number of PCs)');
ylabel('Standard deviation (%)');
title('Std of accuracy and error over 10 repeats');
legend('Location','best');
grid on;

%%
% pick a subject index
i = 1; 

% extract the 3D volume for subject i
vol = squeeze(resid_vol(:,:,:,i));  % 120×144×120

% compute the center slice along each dimension
cx = round(size(vol,1)/2);
cy = round(size(vol,2)/2);
cz = round(size(vol,3)/2);

% plot axial, coronal, sagittal views
figure('Position',[100 100 900 300]);
colormap(gray);

subplot(1,3,1);
imagesc(squeeze(vol(:,:,cz)));
axis image off;
title('Axial (Z)');

subplot(1,3,2);
imagesc(squeeze(vol(:,cy,:))');
axis image off;
title('Coronal (Y)'); 

subplot(1,3,3);
imagesc(squeeze(vol(cx,:,:))');
axis image off;
title('Sagittal (X)');

%%
    % ----- k-NN classifier -----
    % k = 5 (you can change this)
   

rVals      = 10:10:180;          % 10, 20, …, 180
numR       = numel(rVals);
trainAccknn   = zeros(numR,1);      % training accuracies
cvErrknn      = zeros(numR,1); 



r = 150;  % number of prinicpal components (chosen as number of total MRI scans)


for i = 1:numR
    
    r =  rVals(i);

    x_star = PCA_LDA(r,n,m_p, m_c, m, lab, tmp_sampled);
    %disp(x_star)

     K = 5;
    KNNModel = fitcknn(x_star, lab, ...
                       'NumNeighbors', K, ...
                       'Standardize', true, ...
                       'ClassNames', [0 1]);

    % 5-fold cross-validation
    CVKNNModel = crossval(KNNModel, 'KFold', 5);
    cvLoss     = kfoldLoss(CVKNNModel);      % test-fold error

    % training accuracy (on all data)
    labelPred      = predict(KNNModel, x_star);
    trainAcctemp   = mean(labelPred == lab);

    % store
    cvErrknn(i)     = cvLoss;
    trainAccknn(i)  = trainAcctemp;

end 
%%
% ----- k-NN classifier: Sweep over r and K -----

rVals = 10:10:180;           % Number of principal components
numR = numel(rVals);

KVals = 2:2:10;              % Number of neighbors
numK = numel(KVals);

% Pre-allocate result matrices (rows: rVals, cols: KVals)
trainAccknn = zeros(numR, numK);
cvErrknn = zeros(numR, numK);

for i = 1:numR
    r = rVals(i);
    
    % Get reduced features
    x_star = PCA_LDA(r, n, m_p, m_c, m, lab, tmp_sampled);
    
    for j = 1:numK
        K = KVals(j);
        
        % Train k-NN
        KNNModel = fitcknn(x_star, lab, ...
                           'NumNeighbors', K, ...
                           'Standardize', true, ...
                           'ClassNames', [0 1]);
        
        % Cross-validation error
        CVKNNModel = crossval(KNNModel, 'KFold', 5);
        cvLoss = kfoldLoss(CVKNNModel);
        
        % Training accuracy
        labelPred = predict(KNNModel, x_star);
        trainAcc = mean(labelPred == lab);
        
        % Store results
        cvErrknn(i, j) = cvLoss;
        trainAccknn(i, j) = trainAcc;
    end
end
%%
% ----- Plotting -----

% Plot cross-validation error vs. r for different K
figure;
hold on;
for j = 1:numK
    plot(rVals, cvErrknn(:, j), '-o', 'DisplayName', ['K = ', num2str(KVals(j))]);
end
xlabel('Number of Principal Components (r)');
ylabel('Cross-Validation Error');
title('Cross-Validation Error vs. r for Different K');
legend('show');
grid on;
hold off;

% Plot training accuracy vs. r for different K
figure;
hold on;
for j = 1:numK
    plot(rVals, trainAccknn(:, j), '-o', 'DisplayName', ['K = ', num2str(KVals(j))]);
end
xlabel('Number of Principal Components (r)');
ylabel('Training Accuracy');
title('Training Accuracy vs. r for Different K');
legend('show');
grid on;
hold off;


%%
figure
yyaxis left
plot(10:10:180,trainAccknn*100,'-o','LineWidth',1.5, ...
     'DisplayName','Training accuracy')
ylabel('Accuracy (%)')

yyaxis right
plot(10:10:180,cvErrknn*100,'-s','LineWidth',1.5, ...
     'DisplayName','5-fold CV error')
ylabel('CV error (%)')

xlabel('r (number of features / PCs)')
title('KNN performance vs. r')
legend('Location','best')
grid on

%%

figure
plot(10:10:180,trainAccknn*100,'-o','LineWidth',1.5, ...
     'DisplayName','Training accuracy KNN')
hold on
plot(10:10:180,trainAccSVM*100,'-s','LineWidth',1.5,'DisplayName','Training accuracy SVM')

ylabel('Accuracy (%)')
xlabel('r (number of features / PCs)')
title('KNN vs SVM performance ')
legend('Location','best')
grid on
hold off

%%
figure
plot(10:10:180,cvErrknn*100,'-o','LineWidth',1.5, ...
     'DisplayName','5-fold CV error KNN')
hold on
plot(10:10:180,cvErrSVM*100,'-s','LineWidth',1.5,'DisplayName','5-fold CV error SVM')

ylabel('5-fold CV error (%)')
xlabel('r (number of features / PCs)')
title('KNN vs SVM error ')
legend('Location','best')
grid on
hold off
%%


rVals      = 10:10:180;          % 10, 20, …, 180
numR       = numel(rVals);
trainAccNM   = zeros(numR,1);      % training accuracies
cvErrNM      = zeros(numR,1); 



for i = 1:numR
    
    r =  rVals(i);

    x_star = PCA_LDA(r,n,m_p, m_c, m, lab, tmp_sampled);
 

        % ----- Nearest‐mean classifier -----
    % first compute the two class‐means on *all* data
    mu0 = mean( x_star(lab==0) );  
    mu1 = mean( x_star(lab==1) );
    centroids = [mu0; mu1];       % 2×1 vector

    % for 5-fold CV, do a manual partition:
    cvp = cvpartition(lab,'KFold',5);
    errs = zeros(cvp.NumTestSets,1);
    for f = 1:cvp.NumTestSets
        tr = cvp.training(f);
        te = cvp.test(f);
        % recompute centroids on *train* only:
        mu0_tr = mean( x_star(tr & lab==0) );
        mu1_tr = mean( x_star(tr & lab==1) );
        C_tr   = [mu0_tr; mu1_tr];    % 2×1

        % distances of test points to the two centroids
        D = pdist2( x_star(te), C_tr );  % |te|×2
        [~, idx] = min(D,[],2);           % 1 for class-0, 2 for class-1

        yhat = idx-1;                     % convert to {0,1}
        errs(f) = mean( yhat ~= lab(te) );
    end
    cvLoss    = mean(errs);               % averaged CV error
    
    % training accuracy on full data
    D_all = pdist2( x_star, centroids );
    [~, idxAll]  = min(D_all,[],2);
    labelPred    = idxAll - 1;
    trainAcctemp = mean(labelPred == lab);

    % store
    cvErrNM(i)     = cvLoss;
    trainAccNM(i)  = trainAcctemp;


end 
%%

figure
plot(10:10:180,trainAccknn*100,'-o','LineWidth',1.5, ...
     'DisplayName','Training accuracy KNN')
hold on
plot(10:10:180,trainAccSVM*100,'-s','LineWidth',1.5,'DisplayName','Training accuracy SVM')
plot(10:10:180,trainAccNM*100,'-d','LineWidth',1.5,'DisplayName','Training accuracy Nearest Mean')

ylabel('Accuracy (%)')
xlabel('r (number of features / PCs)')
title('KNN vs SVM vs NM performance ')
legend('Location','best')
grid on
hold off
%%
figure
plot(10:10:180,cvErrknn*100,'-o','LineWidth',1.5, ...
     'DisplayName','5-fold CV error KNN')
hold on
plot(10:10:180,cvErrSVM*100,'-s','LineWidth',1.5,'DisplayName','5-fold CV error SVM')
plot(10:10:180,cvErrNM*100,'-d','LineWidth',1.5,'DisplayName','5-fold CV error Nearest Mean')

ylabel('5-fold CV error (%)')
xlabel('r (number of features / PCs)')
title('KNN vs SVM vs NM error ')
legend('Location','best')
grid on
hold off