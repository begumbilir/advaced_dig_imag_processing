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
% clear all
% close all

%datadir='E:\data\oasis\' 

load oasis_residual_dataset_subs_05_20150309T105732_19924
%load oasis_dataset_subs_05_20150309T105732_19924


%% prepare data
sz = size(resid_vol); % size of data
lab = stats.CDR*2;    % get group label: 0 is controls, 1 is MCI patients

age = stats.Age;

% Sample the input data (initial sizes are very large)
resid_vol_sampled = resid_vol(1:3:end, 1:3:end, 1:3:end, :);

% apply mask, get one matrix with nvoxels x nsubjects
sz_sampled = size(resid_vol_sampled);
tmp_sampled = reshape(resid_vol_sampled,prod(sz_sampled(1:3)),[]);

m_p = sum(lab > 0); % number of patients
m_c = sum(lab == 0); % number of controls
m = m_p + m_c; %total number of participants
n = length(tmp_sampled); 

tmp_p = tmp_sampled(:,lab > 0); % vectorized voxels for MRI scans of Alzhemier patients
tmp_c = tmp_sampled(:, lab == 0); % vectorized voxels for MRI scans of healthy (control) group

% Chose r
r = 100;

%% Apply singular value decomposition

%[P_mtrx, sing_vals] = svd(tmp_sampled_selected, 'econ');
[U, S, ~] = svd(tmp_sampled, 'econ');
P_mtrx_r = U(:, 1:r);

% Reverse columns of P_mtrx_r manually(eigenvalues are ordered in monotonically increased)
P_mtrx = P_mtrx_r(:, end:-1:1);

%% Create Scattered matrix and Project x_i onto P matrix
Scattered_P = zeros(n,n);
tmp_prime = zeros(r, m);
for i = 1:m
    x_i = tmp_sampled(:,i);
    x_mean = mean(tmp_sampled, 2);  % mean along rows
    x_i_centered = x_i - x_mean;
    % Create Scattered matrix
    Scattered_P = Scattered_P + x_i_centered * transpose(x_i_centered);
    % Projecting x_i onto P matrix
    x_i_prime = transpose(P_mtrx) * x_i_centered;
    tmp_prime(:,i) = x_i_prime; %size r x 1
end


%% Create q using LDA

x_p_prime_mean = zeros(r ,1);
x_c_prime_mean = zeros(r ,1);
for i = 1:m
    x_i_prime = tmp_prime(:,i); %size rx1
    if lab(i) == 1 %if patient
        x_p_prime_mean = x_p_prime_mean + x_i_prime;
    else %if control
        x_c_prime_mean = x_c_prime_mean + x_i_prime;
    end     
end

x_p_prime_mean = x_p_prime_mean ./ m_p;
x_c_prime_mean = x_c_prime_mean ./ m_c;

x_prime_mean = (m_p .* x_p_prime_mean + m_c .* x_c_prime_mean) ./ m; 


%% Creating Sw and Sb 

sw_p = zeros(r,r);
sw_c = zeros(r,r);
for i = 1:m
   x_i_prime = tmp_prime(:,i);
   if lab(i) == 1 %if patient
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
%%
%create q vector and a vector 

q_mtrx = inv(Scattered_w)*Scattered_b;
[W, D] = eig(q_mtrx);
[eigvals, order] = sort(diag(D), 'descend');  % sort eigenvalues largest → smallest
q_leading = W(:, order(1)); % leading eigenvector (r×1)
a = P_mtrx*q_leading; 
%% Finding x*
x_star = zeros(r,1);
for i = 1:m
    x_i = tmp_sampled(:,i);
    x_mean = mean(tmp_sampled, 2);  % mean along rows
    x_star_i = transpose(a)*(x_i-x_mean);
    x_star(i,1) = x_star_i;
end

%%
%classification with svm and 5 fold cross validation 

% --- Example data placeholders ---
% X = randn(182, r);           % your 182×r feature matrix
% Y = [ones(60,1); zeros(122,1)];  % 1=patient (60), 0=healthy (122)
% -----------------------------------

% (1) Train a linear SVM, with automatic feature standardization
SVMModel = fitcsvm(x_star, lab, ...
    'KernelFunction', 'linear', ...
    'Standardize',     true, ...
    'ClassNames',      [0 1]);

% (2) (Optional) Estimate out-of-sample loss via 5-fold cross-validation
CVSVMModel = crossval(SVMModel, 'KFold', 5);
cvLoss = kfoldLoss(CVSVMModel);
fprintf('5-fold CV classification error: %.2f%%\n', cvLoss*100);

% (3) Predict on your (training) data (or on a held-out test set)
[labelPred, scorePred] = predict(SVMModel, x_star);

% (4) Compute training accuracy
trainAcc = mean(labelPred == lab);
fprintf('Training accuracy: %.2f%%\n', trainAcc*100);



%% Printing the figure of the input data and its quantized version

% pick a subject index
i = 10; 

% --- Initial data ---
vol1 = squeeze(resid_vol(:,:,:,i));  % 60×72×60
cx1 = round(size(vol1,1)/2);
cy1 = round(size(vol1,2)/2);
cz1 = round(size(vol1,3)/2);

% --- Quantized data ---
vol2 = squeeze(resid_vol_sampled(:,:,:,i));  % 20×24×20
cx2 = round(size(vol2,1)/2);
cy2 = round(size(vol2,2)/2);
cz2 = round(size(vol2,3)/2);

% --- PLOT ---
figure('Position',[100 100 900 600]);
colormap(gray);

% Row 1: resid_vol
subplot(2,3,1);
imagesc(squeeze(vol1(:,:,cz1)));
axis image off;
title('Original Axial (Z)');

subplot(2,3,2);
imagesc(squeeze(vol1(:,cy1,:))');
axis image off;
title('Original Coronal (Y)');

subplot(2,3,3);
imagesc(squeeze(vol1(cx1,:,:))');
axis image off;
title('Original Sagittal (X)');

% Row 2: resid_vol_sampled
subplot(2,3,4);
imagesc(squeeze(vol2(:,:,cz2)));
axis image off;
title('Sampled Axial (Z)');

subplot(2,3,5);
imagesc(squeeze(vol2(:,cy2,:))');
axis image off;
title('Sampled Coronal (Y)');

subplot(2,3,6);
imagesc(squeeze(vol2(cx2,:,:))');
axis image off;
title('Sampled Sagittal (X)');

%% Reconstruct the images after PCA
reconst_tmp_sampled = zeros(n,m);
for i = 1:m
    y_i = tmp_prime(:,i); %size rx1
    x_mean = mean(tmp_sampled, 2);  % mean along rows
    % P*y_i + x_mean
    % Projecting x_i onto P matrix
    x_i_reconst = P_mtrx * y_i + x_mean;
    reconst_tmp_sampled(:,i) = x_i_reconst;
end

resid_vol_sampled_reconstructed = reshape(reconst_tmp_sampled, sz_sampled);
%Upsample the reconstructed image and the sampled original image
sz_original = sz_sampled .* [3, 3, 3, 1];
resid_vol_upsampled = zeros(sz_original);
resid_vol_upsampled_reconstructed = zeros(sz_original);

for t = 1:sz_sampled(4)
    resid_vol_upsampled(:,:,:,t) = imresize3(resid_vol_sampled(:,:,:,t), sz_original(1:3));
    resid_vol_upsampled_reconstructed(:,:,:,t) = imresize3(resid_vol_sampled_reconstructed(:,:,:,t), sz_original(1:3));
end


%% Plot together with the error between the upsampled initial image and upsampled reconstructed
% Pick a subject index
i = 10;

% --- Original data ---
vol1 = squeeze(resid_vol_upsampled(:,:,:,i));  % 20×24×20
cx1 = round(size(vol1,1)/2);
cy1 = round(size(vol1,2)/2);
cz1 = round(size(vol1,3)/2);

% --- Reconstructed data ---
vol2 = squeeze(resid_vol_upsampled_reconstructed(:,:,:,i));  % 20×24×20
cx2 = round(size(vol2,1)/2);
cy2 = round(size(vol2,2)/2);
cz2 = round(size(vol2,3)/2);

% --- Compute error (absolute difference) ---
error_vol = abs(vol1 - vol2);
cx_err = round(size(error_vol,1)/2);
cy_err = round(size(error_vol,2)/2);
cz_err = round(size(error_vol,3)/2);

% --- PLOT ---
figure('Position',[100 100 1000 900]);  % make taller to fit 3 rows
colormap(gray);

% Row 1: Original
subplot(3,3,1);
imagesc(squeeze(vol1(:,:,cz1)));
axis image off;
title('Original Axial (Z)');

subplot(3,3,2);
imagesc(squeeze(vol1(:,cy1,:))');
axis image off;
title('Original Coronal (Y)');

subplot(3,3,3);
imagesc(squeeze(vol1(cx1,:,:))');
axis image off;
title('Original Sagittal (X)');

% Row 2: Reconstructed
subplot(3,3,4);
imagesc(squeeze(vol2(:,:,cz2)));
axis image off;
title('Reconstructed Axial (Z)');

subplot(3,3,5);
imagesc(squeeze(vol2(:,cy2,:))');
axis image off;
title('Reconstructed Coronal (Y)');

subplot(3,3,6);
imagesc(squeeze(vol2(cx2,:,:))');
axis image off;
title('Reconstructed Sagittal (X)');

% Row 3: Error (absolute difference)
subplot(3,3,7);
imagesc(squeeze(error_vol(:,:,cz_err)));
axis image off;
title('Error Axial (Z)');

subplot(3,3,8);
imagesc(squeeze(error_vol(:,cy_err,:))');
axis image off;
title('Error Coronal (Y)');

subplot(3,3,9);
imagesc(squeeze(error_vol(cx_err,:,:))');
axis image off;
title('Error Sagittal (X)');

%sgtitle(['Subject ' num2str(i) ' - Original, Reconstructed, and Error']);

mse_error = mean((vol1(:) - vol2(:)).^2);
fprintf('MSE Error: %.6f\n', mse_error);
