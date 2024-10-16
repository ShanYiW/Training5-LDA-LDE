% 5折交叉验证
clear;
% mode = 'area';
% load(append("ORL_14_", mode)); % vector-LDE 用1/4下采样的ORL
% DATA = double(ORL); clear ORL;
load('Yale_32.mat');
DATA = fea'; clear fea; % X:D*N  gnd:N*1

[D,N] = size(DATA);
rand('seed', 6); % 
cv5 = randperm(N)';
fold = 5; % 5-fold cross validation
Ntest = floor(N/fold);
Ntrain = N - Ntest;
%% 遍历(d1, d2), 固定(k, t).
acc = 0;
for f=1:fold
    is_test = false([N,1]);
    is_test((f-1)*Ntest+1:f*Ntest) = true;
    is_train = ~is_test;
    
    testset = DATA(:, cv5(is_test)); % D*Ntest
    gndtest = gnd(cv5(is_test)); % Ntest*1
    trainset = DATA(:, cv5(is_train)); % D*Ntrain
    gndtrain = gnd(cv5(is_train)); % Ntrain*1
    
    train2 = sum(trainset.*trainset, 1); % 1*Ntrain
    test2 = sum(testset.*testset, 1); % 1*Ntest
    dis_trte = repmat(train2', [1,Ntest]) + repmat(test2, [Ntrain,1]) - 2.*trainset'*testset; % Ntrain*Ntest
    
    [~, idx] = sort(dis_trte); % 
    nearest_nei = idx(1,:); % 1*Ntest
    acc = acc + sum(gndtrain(nearest_nei)==gndtest);
end
acc = acc/(fold*Ntest);
fprintf('Acc=%f\n', acc);