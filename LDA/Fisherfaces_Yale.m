clear;
load('Yale_64.mat');
Yale = fea'; clear fea; % X:D*N  gnd:N*1
[D,N] = size(Yale);
rand('seed', 6); % 
cv5 = randperm(N)';
fold = N; % 5-fold cross validation
Ntest = floor(N/fold);
%%
lb_d=2; ub_d=60; len_d = ub_d-lb_d+1;
Acc_cv5 = zeros([fold, len_d]);
time0 = clock;
for f=1:fold
    is_test = false([N,1]);
    is_test((f-1)*Ntest+1:f*Ntest) = true;
    is_train = ~is_test;
    
    testset = Yale(:, cv5(is_test)); % D*Ntest
    gndtest = gnd(cv5(is_test)); % Ntest*1
    trainset = Yale(:, cv5(is_train)); % D*Ntrain
    gndtrain = gnd(cv5(is_train)); % Ntrain*1
    
    [E] = Fisherfaces(trainset, gndtrain); % E:D*D
    for d=lb_d:ub_d
        Y_train = E(:,1:d)'*trainset; % 1:d
        Y_test  = E(:,1:d)'*testset; % 2:d+1
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
Time_used = etime(clock, time0);
fprintf('Time used: %f s\n', Time_used);
Acc_avg = mean(Acc_cv5)'; % len_d*1
maxAcc = Acc_avg(1); maxidx = 1;
for j=2:len_d
    if Acc_avg(j)>maxAcc
        maxAcc = Acc_avg(j);
        maxidx = j;
    end
end
fprintf('Max: %f %d\n', maxAcc, maxidx);
figure;
plot(lb_d:ub_d, Acc_avg, '^-', 'Color',[162,20,47]./255,'MarkerFaceColor','w','Linewidth',1.5);
xlabel('Dims', 'Fontsize', 16);
ylabel('Recognition rate (%)', 'Fontsize', 16);

