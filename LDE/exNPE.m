function [E] = exNPE(X, gnd, ki, ko)
% X: D*N
% gnd: N*1 label标签
% ki, ko: 近邻数
% E: D*N (假设 d<N, d<D)

N = length(gnd); % 样本数
cls_label = unique(gnd); % 类别标签
c = length(cls_label); % 类别数
%% PCA降维 预处理 (效果↑, 能稳定在98.75%)
[Wpca] = PCA_DR(X, 0.975); % D*r  r<D
X = Wpca'*X; % r*N
%%
X2 = sum(X.*X, 1); % 1*Nc
dist = repmat(X2, [N,1]) + repmat(X2', [1,N]) - 2.*X'*X; % N*N
%% 对于每一类, 构造其与同类样本的邻接矩阵, 其与异类样本的邻接矩阵
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
        Adj_o(glb_idx_o(nei_idx_o(:,k)), glb_idx_i(k)) = true;
    end
end
%% 构造 加权的邻接矩阵Wi, Wo
Wi = zeros([N,N]); Wo = zeros([N,N]);
tol = 1e-12;
for j=1:N
    k = min(ki, sum(Adj_i(:,j)));
    Xj = repmat(X(:,j), [1,k]) - X(:, Adj_i(:,j)); % D*ki
    invG = inv( Xj'*Xj + tol*sum(sum(Xj.^2)).*eye(k) ); % ki*ki, 当K>D时, 警告：矩阵接近奇异值，或者缩放错误。结果可能不准确。
    w = sum(invG, 2) ./ sum(sum(invG)); % k*1
    Wi(Adj_i(:,j), j) = w;
end
for j=1:N
    Xj = repmat(X(:,j), [1,ko]) - X(:, Adj_o(:,j)); % D*ki
    invG = inv( Xj'*Xj + tol*sum(sum(Xj.^2)).*eye(ko) ); % ko*ko, 当K>D时, 警告：矩阵接近奇异值，或者缩放错误。结果可能不准确。
    w = sum(invG, 2) ./ sum(sum(invG)); % k*1
    Wo(Adj_o(:,j), j) = w;
end
%% 
[U,S,~] = svd(X*(eye(N) - Wi)); % U:D*D  S:D*N
[row, col] = size(S); 
S = diag(S);
invS = zeros([col, row]);
if row > col
    invS(:,1:col) = diag(1./S); % col*col
else % row <= col
    invS(1:row, :) = diag(1./S); % row*row
end
M = invS*U'*X*(eye(N) - Wo); % N*N
[Evec, ~, ~] = svd(M); 
E = U*invS'*Evec; % D*N
E = Wpca*E; % D*r * r*N = D*N

return;