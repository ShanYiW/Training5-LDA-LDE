function [L,R] = d2LDE(X, D1,D2, gnd, ki, ko, t)
% X: D*N 数据. D:维数  N:样本数
% gnd: N*1. 类别标签
% ki: 类内近邻数  ko: 属于其他类的近邻数
% t: 热核函数的参数  
% L: D1*D1  R:D2*D2
cls_label = unique(gnd); % 类别标签
c = length(cls_label); % 类别数
%% 构造Wi, Wo \in N*N
N = length(gnd);
X2 = sum(X.*X, 1); % 1*Nc
dist = repmat(X2, [N,1]) + repmat(X2', [1,N]) - 2.*X'*X; % N*N
Adj_i = false([N,N]); % 同类样本的邻接矩阵
Adj_o = false([N,N]); % 异类样本的邻接矩阵
for i=1:c
    Xi_idx = gnd==cls_label(i); % N*1, T/F 第i类的样本的编号
    Xo_idx = ~Xi_idx; % 不是第i类的样本编号
    Ni = length(gnd(Xi_idx)); % 第i类样本数
    No = N - Ni; % 不是第i类的样本数
    
    glb_idx_i = zeros([Ni,1]); glb_idx_o = zeros([No,1]);
    j=1; % 构造 第i类 类内编号->全局编号的映射 glb_idx
    for k=1:Ni
        while ~Xi_idx(j) % Xc_idx(k) == 0
            j = j + 1;
        end
        glb_idx_i(k) = j;
        j = j + 1;
    end
    j=1; % 构造 非第i类编号->全局编号的映射 glb_idx
    for k=1:No
        while ~Xo_idx(j) % Xc_idx(k) == 0
            j = j + 1;
        end
        glb_idx_o(k) = j;
        j = j + 1;
    end
    dist_i = dist(Xi_idx, Xi_idx); % Ni*Ni
    [~, nei_idx_i] = sort(dist_i); % Ni*Ni
    nei_idx_i = nei_idx_i(2:min(ki+1, Ni),:); % ki*Ni 不包括自己 若类内样本不足ki+1, 则全做邻居
    dist_o = dist(Xo_idx, Xi_idx); % No*Ni
    [~, nei_idx_o] = sort(dist_o); % No*Ni
    nei_idx_o = nei_idx_o(1:min(ko, No),:); % ko*Ni
    for k=1:Ni % 对于第i类的每个样本
        Adj_i(glb_idx_i(nei_idx_i(:,k)), glb_idx_i(k)) = true;
        Adj_i(glb_idx_i(k), glb_idx_i(nei_idx_i(:,k))) = true;
        Adj_o(glb_idx_o(nei_idx_o(:,k)), glb_idx_i(k)) = true;
        Adj_o(glb_idx_i(k), glb_idx_o(nei_idx_o(:,k))) = true;
    end
end
Wi = zeros([N,N]); Wo = zeros([N,N]);
Wi(Adj_i) = exp(-1.*dist(Adj_i)./(4*t)); % N*N
Wo(Adj_o) = exp(-1.*dist(Adj_o)./(4*t)); % N*N
%% 求 L, R
X3 = zeros([D1, D2, N]); % 每个样本还原成矩阵
for j=1:N
    X3(:,:,j) = reshape(X(:, j), [D1,D2]); % D*1 -> D1*D2, 列优先堆叠
end

L = zeros([D1,D1]); R = zeros([D2,D2]);
converged=false; ratio=Inf; it=1;
while ~converged
    L_old = L;
    % 更新R: D2*D2
    MLo = zeros([D2,D2]); 
    MLi = zeros([D2,D2]);
    for j=1:N
        for i=1:N
            Lt_Xi_Xj = L*(X3(:,:,i)-X3(:,:,j)); % D1*D2
            MLo = MLo + Lt_Xi_Xj'*Lt_Xi_Xj .* Wo(i,j); % D2*D2
            MLi = MLi + Lt_Xi_Xj'*Lt_Xi_Xj .* Wi(i,j); % D2*D2
        end
    end
    [Evec, ~] = eig(MLo, MLi); % Evec, Eval: D2*D2  Eval=diag(Eval);
    R = Evec(:,end:-1:1);
    % 更新L: D1*D1
    MRo = zeros([D1, D1]);
    MRi = zeros([D1, D1]);
    for j=1:N
        for i=1:N
            Xi_Xj_R = (X3(:,:,i)-X3(:,:,j))*R; % D1*D2
            MRo = MRo + Xi_Xj_R*Xi_Xj_R' .* Wo(i,j); % D1*D1
            MRi = MRi + Xi_Xj_R*Xi_Xj_R' .* Wi(i,j); % D1*D1
        end
    end
    [Evec, ~] = eig(MRo, MRi); % Evec: D1*D1
    L = Evec(:,end:-1:1); % L: D1*D1
    % 判敛
    LLp_fro2 = sum(sum((L_old-L).*(L_old-L)));
    L_fro2 = sum(sum(L.*L));
    ratio_old = ratio;
    ratio = LLp_fro2/L_fro2;
    if abs(ratio-ratio_old) < 1e-2 || it>10; converged=true; end
    it = it+1;
end

return