% 5折交叉验证
clear;
mode = 'area';
load(append("ORL_14_", mode)); % vector-LDE 用1/4下采样的ORL
ORL = double(ORL);
[D,N] = size(ORL);
rand('seed', 6); % 6,3, 70, 2.9e11, 975
cv5 = randperm(N)';
fold = 5; % 5-fold cross validation
Ntest = floor(N/fold);
Ntrain = N - Ntest;
%%
k=60; t=2e5; sig2 = 1e6;
lb_d = 2; ub_d = 60; len_d = ub_d-lb_d+1;
Acc_cv5 = zeros([fold, len_d]);
for f=1:fold
    is_test = false([N,1]);
    is_test((f-1)*Ntest+1:f*Ntest) = true;
    is_train = ~is_test;
    
    testset = ORL(:, cv5(is_test)); % D*Ntest
    gndtest = gnd(cv5(is_test)); % Ntest*1
    trainset = ORL(:, cv5(is_train)); % D*Ntrain
    gndtrain = gnd(cv5(is_train)); % Ntrain*1
    
    train2 = sum(trainset.*trainset, 1); % 1*Ntrain
    dis_tr = repmat(train2, [Ntrain,1]) + repmat(train2', [1,Ntrain]) - 2.*trainset'*trainset; % Ntrain*Ntrain
    test2 = sum(testset.*testset, 1); % 1*Ntest
    dis_trte = repmat(train2', [1,Ntest]) + repmat(test2, [Ntrain,1]) - 2.*trainset'*testset; % Ntrain*Ntest
    
    [E] = kLPP(trainset, k, t, sig2); % E:Ntrain * Ntrain
    for d=lb_d:ub_d
        Y_train = E(:,1:d)'*exp(-0.5/sig2 .*dis_tr); % (d*Ntr) * (Ntr*Ntr) = d*Ntr
        Y_test  = E(:,1:d)'*exp(-0.5/sig2 .*dis_trte); % (d*Ntr) * (Ntr*Nte) = d*Nte
        acc = 0;
        for j=1:Ntest
            y = Y_test(:,j); % d*1
            dist = sum((repmat(y,[1,N-Ntest]) - Y_train).^2, 1);
            [~,idx] = sort(dist); % 距离 升序排
            pred = gndtrain(idx(1)); % 最近邻的类别 = j号测试样本的类别
            acc = acc + (pred==gndtest(j));
        end
        acc = acc/Ntest; % 
        Acc_cv5(f, d-lb_d+1) = acc;
    end
end
Acc_avg = mean(Acc_cv5)'; % len_d*1
maxAcc = Acc_avg(1); maxidx = 1;
for j=2:len_d
    if Acc_avg(j)>maxAcc
        maxAcc = Acc_avg(j);
        maxidx = j;
    end
end
fprintf('Max: %f %d\n', maxAcc, maxidx);
figure('Name',['k',num2str(k), '_t',num2str(t)]);
plot(lb_d:ub_d, Acc_avg, '^-', 'Color',[162,20,47]./255, 'MarkerFaceColor','w','Linewidth', 1.5);
xlabel('Dims', 'Fontsize', 16);
ylabel('Recognition rate (%)', 'Fontsize', 16);
%% 遍历 k, sig2
% t = 10e4;
% lb_sig2=1e6; ub_sig2=10e6; step_sig2=5e5; len_sig2=floor((ub_sig2-lb_sig2)/step_sig2)+1;
% lb_k = 2; ub_k = 60; s_k=2; len_k = floor((ub_k-lb_k)/s_k)+1;
% d = ub_k;
% Acc_tk = zeros([len_sig2, len_k, fold]);
% for f=1:fold
%     is_test = false([N,1]);
%     is_test((f-1)*Ntest+1:f*Ntest) = true;
%     is_train = ~is_test;
%     
%     testset = ORL(:, cv5(is_test)); % D*Ntest
%     gndtest = gnd(cv5(is_test)); % Ntest*1
%     trainset = ORL(:, cv5(is_train)); % D*Ntrain
%     gndtrain = gnd(cv5(is_train)); % Ntrain*1
% 
%     train2 = sum(trainset.*trainset, 1); % 1*Ntrain
%     dis_tr = repmat(train2, [Ntrain,1]) + repmat(train2', [1,Ntrain]) - 2.*trainset'*trainset; % Ntrain*Ntrain
%     test2 = sum(testset.*testset, 1); % 1*Ntest
%     dis_trte = repmat(train2', [1,Ntest]) + repmat(test2, [Ntrain,1]) - 2.*trainset'*testset; % Ntrain*Ntest
% 
%     for sig2=lb_sig2:step_sig2:ub_sig2
%         for k=lb_k:s_k:ub_k
%             [E] = kLPP(trainset, k, t, sig2); % E:Ntrain * Ntrain
% 
%             Y_train = E(:,1:d)'*exp(-0.5/sig2 .*dis_tr); % (d*Ntr) * (Ntr*Ntr) = d*Ntr
%             Y_test  = E(:,1:d)'*exp(-0.5/sig2 .*dis_trte); % (d*Ntr) * (Ntr*Nte) = d*Nte
%             acc = 0;
%             for j=1:Ntest
%                 y = Y_test(:,j); % d*1
%                 dist = sum((repmat(y,[1,N-Ntest]) - Y_train).^2, 1);
%                 [~,idx] = sort(dist); % 距离 升序排
%                 pred = gndtrain(idx(1)); % 最近邻的类别 = j号测试样本的类别
%                 acc = acc + (pred==gndtest(j));
%             end
%             acc = acc/Ntest; % 
%             Acc_tk((sig2-lb_sig2)/step_sig2+1, (k-lb_k)/s_k+1, f) = acc;
%         end
%     end
% end
% % 多次交叉验证的平均
% Acc_avg = mean(Acc_tk, 3); % len_t, len_k
% 
% figure;
% bar_3 = bar3(Acc_avg);
% for k=1:length(bar_3)
%     bar_3(k).CData = bar_3(k).ZData; % 颜色与Z取值成正比
%     bar_3(k).FaceColor = 'interp';
% end
% zlim([0.85,1]);
% set(gca, 'xTick', [1:2:len_k]); % ko
% set(gca, 'xTicklabel', split(num2str( lb_k:2*s_k:ub_k )) ); % 不能split(, ' '), 因为'2   3'-> {'2'},{' '},{' '},{'3'}
% set(gca, 'yTick', [1:len_sig2], 'Fontsize',8); % 
% set(gca, 'yTicklabel', split(num2str( lb_sig2:step_sig2:ub_sig2 )), 'yTickLabelRotation', -45 );
% set(gca,'XGrid', 'off', 'YGrid', 'off','ZGrid', 'on');
% xlabel('$k$', 'Interpreter', 'latex', 'Fontsize', 14);
% ylabel('$\sigma^2$', 'Interpreter', 'latex', 'Fontsize', 14);
% zlabel('$Accuracy$', 'Interpreter', 'latex', 'Fontsize', 14);
% view([-90, 1]); % 0度:沿第二维度(↓)的反方向(↑) 正视第一维度(→)