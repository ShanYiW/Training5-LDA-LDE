% 5折交叉验证
clear;
load('Yale_32.mat');
DATA = fea'; clear fea; % X:D*N  gnd:N*1
[D,N] = size(DATA);
rand('seed', 6); % 6,3, 70, 2.9e11, 975
cv5 = randperm(N)';
fold = 5; % 5-fold cross validation
Ntest = floor(N/fold); % 测试样本数
Ntrain = N - Ntest; % 训练样本数
%%
ki=5; ko=10; t=1e7; sig2=23e6;
lb_d = 2; ub_d = 60; len_d = ub_d-lb_d+1;
Acc_cv5 = zeros([fold, len_d]);
for f=1:fold
    is_test = false([N,1]);
    is_test((f-1)*Ntest+1:f*Ntest) = true;
    is_train = ~is_test;
    
    testset = DATA(:, cv5(is_test)); % D*Ntest
    gndtest = gnd(cv5(is_test)); % Ntest*1
    trainset = DATA(:, cv5(is_train)); % D*Ntrain
    gndtrain = gnd(cv5(is_train)); % Ntrain*1
    
    train2 = sum(trainset.*trainset, 1); % 1*Ntrain
    dis_tr = repmat(train2, [Ntrain,1]) + repmat(train2', [1,Ntrain]) - 2.*trainset'*trainset; % Ntrain*Ntrain
    test2 = sum(testset.*testset, 1); % 1*Ntest
    dis_trte = repmat(train2', [1,Ntest]) + repmat(test2, [Ntrain,1]) - 2.*trainset'*testset; % Ntrain*Ntest
    
    [E] = kLDE(trainset, gndtrain, ki, ko, t, sig2); % E:N*N
    for d=lb_d:ub_d
        Y_train = E(:,1:d)'*exp(-0.5/sig2 .*dis_tr); % (d*Ntr) * (Ntr*Ntr) = d*Ntr
        Y_test  = E(:,1:d)'*exp(-0.5/sig2 .*dis_trte); % (d*Ntr) * (Ntr*Nte) = d*Nte
        acc = 0;
        for j=1:Ntest
            y = Y_test(:,j); % d*1
            dist = sum((repmat(y,[1,Ntrain]) - Y_train).^2, 1);
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
figure('Name',['ki',num2str(ki),'_ko',num2str(ko), '_t',num2str(t)]);
plot(lb_d:ub_d, Acc_avg, '^-', 'Color',[162,20,47]./255, 'MarkerFaceColor','w','Linewidth', 1.5);
xlabel('Dims', 'Fontsize', 16);
ylabel('Recognition rate (%)', 'Fontsize', 16);
%% 遍历 ki, t, sig2
% for sig2 = [23e6]
% ko = 10;
% t_arr = [1e5 3e5 1e6 3e6 1e7 3e7];
% len_t = length(t_arr);
% l_ki = 2; u_ki = 9; len_ki = u_ki - l_ki + 1;
% Acc_tki = zeros([len_t, len_ki, fold]);
% d = 60;
% for f=1:fold
%     is_test = false([N,1]);
%     is_test((f-1)*Ntest+1:f*Ntest) = true;
%     is_train = ~is_test;
%     
%     testset = DATA(:, cv5(is_test)); % D*Ntest
%     gndtest = gnd(cv5(is_test)); % Ntest*1
%     trainset = DATA(:, cv5(is_train)); % D*Ntrain
%     gndtrain = gnd(cv5(is_train)); % Ntrain*1
%     train2 = sum(trainset.*trainset, 1); % 1*Ntrain
%     dis_tr = repmat(train2, [Ntrain,1]) + repmat(train2', [1,Ntrain]) - 2.*trainset'*trainset; % Ntrain*Ntrain
%     test2 = sum(testset.*testset, 1); % 1*Ntest
%     dis_trte = repmat(train2', [1,Ntest]) + repmat(test2, [Ntrain,1]) - 2.*trainset'*testset; % Ntrain*Ntest
% 
% for tid = 1:len_t
%     for ki=l_ki:u_ki
%         [E] = kLDE(trainset, gndtrain, ki, ko, t_arr(tid), sig2); % E:D*D
% 
%         Y_train = E(:,1:d)'*exp(-0.5/sig2 .*dis_tr); % (d*Ntr) * (Ntr*Ntr) = d*Ntr
%         Y_test  = E(:,1:d)'*exp(-0.5/sig2 .*dis_trte); % (d*Ntr) * (Ntr*Nte) = d*Nte
%         acc = 0;
%         for j=1:Ntest
%             y = Y_test(:,j); % d*1
%             dist = sum((repmat(y,[1,N-Ntest]) - Y_train).^2, 1);
%             [~,idx] = sort(dist); % 距离 升序排
%             pred = gndtrain(idx(1)); % 最近邻的类别 = j号测试样本的类别
%             acc = acc + (pred==gndtest(j));
%         end
%         acc = acc/Ntest; % 
%         Acc_tki(ki-l_ki+1, tid, f) = acc;
%     end
% end
% end
% Acc_avg = mean(Acc_tki, 3); % len_ki, len_ko
% 
% figure('Name', ['sig2=',num2str(sig2)]);
% bar_3 = bar3(Acc_avg);
% for k=1:length(bar_3)
%     bar_3(k).CData = bar_3(k).ZData; % 颜色与Z取值成正比
%     bar_3(k).FaceColor = 'interp';
% end
% zlim([0.7,0.85]);
% set(gca, 'xTick', [1:len_t]); % ki
% set(gca, 'xTicklabel', split(num2str( 1:len_t )) );
% set(gca, 'yTick', [1:len_ki]); % ki
% set(gca, 'yTicklabel', split(num2str( l_ki:u_ki )) ); % 不能split(, ' '), 因为'2   3'-> {'2'},{' '},{' '},{'3'}
% set(gca,'XGrid', 'off', 'YGrid', 'off','ZGrid', 'on')
% xlabel('$t$', 'Interpreter', 'latex', 'Fontsize', 14);
% ylabel('$k_i$', 'Interpreter', 'latex', 'Fontsize', 14);
% zlabel('$Accuracy$', 'Interpreter', 'latex', 'Fontsize', 14);
% view([-90, 1]); % 0度:延第二维度(↓)的反方向(↑) 正视第一维度(→)
% end