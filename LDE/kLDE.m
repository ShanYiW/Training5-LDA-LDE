function [E] = kLDE(X, gnd, ki, ko, t, sig2)
% LDE+高斯核
% X: D*N 数据. D:维数  N:样本数
% gnd: N*1. 类别标签
% ki: 类内近邻数  ko: 属于其他类的近邻数
% t: 热核函数的参数  sig: 高斯核函数的带宽
% E: D*D 投影
cls_label = unique(gnd); % 类别标签
c = length(cls_label); % 类别数
%% PCA降维 预处理 (效果↑, LDE稳定在98.75%)
% [Wpca] = PCA_DR(X, 0.98); % D*r  r<D
% X = Wpca'*X; % r*N
%% 
N = length(gnd); % 样本数
X2 = sum(X.*X, 1); % 1*Nc
dist = repmat(X2, [N,1]) + repmat(X2', [1,N]) - 2.*X'*X; % N*N
Kernel = exp(-0.5/sig2 .* dist); % N*N  高斯核
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
        Adj_i(glb_idx_i(k), glb_idx_i(nei_idx_i(:,k))) = true;
        Adj_o(glb_idx_o(nei_idx_o(:,k)), glb_idx_i(k)) = true;
        Adj_o(glb_idx_i(k), glb_idx_o(nei_idx_o(:,k))) = true;
    end
end
%% 构造 加权的邻接矩阵Wi, Wo, 构造拉普拉斯Li, Lo
Wi = zeros([N,N]); Wo = zeros([N,N]);
Wi(Adj_i) = exp(-1.*dist(Adj_i)./(4*t)); % N*N
Wo(Adj_o) = exp(-1.*dist(Adj_o)./(4*t)); % N*N
Di = sum(Wi, 2); % N*1
Do = sum(Wo, 2); 
Li = diag(Di) - Wi; % N*N 类内 rank(Li)=N-c, 因为有c个连通分量
Lo = diag(Do) - Wo; % 类间  Lo正定  rank(Lo)=N-1
KLiK = Kernel*Li*Kernel; % N*N
KLiK = (KLiK+KLiK')./2;%max(XLiXt, XLiXt');
KLoK = Kernel*Lo*Kernel; % N*N 
KLoK = (KLoK+KLoK')./2;%max(XLoXt, XLoXt');
%% 直接eig(广义特征值分解)会出虚特征值 或 Inf特征值, 必须用trick
% % [E, Eval] = eig(KLoK, KLiK); % Evec: N*N
%% max Tr(PX Lo X'P'), s.t. PX Li X'P'=I
[Q,S] = eig(KLiK); % Q,S: N*N
S = diag(S); % 默认升序
Negtive = S<0; 
lenNeg = sum(Negtive);
S(Negtive) = -S(lenNeg+1); % +1
invS = diag(1./sqrt(S)); 
M = invS*Q'*KLoK*Q*invS';
M = (M+M')./2;%max(M,M'); % N*N
[Evec, Eval] = eig(M);
Eval = diag(Eval);
[~,idx] = sort(-Eval); % 降序
Evec = Evec(:, idx); 
E = Q*invS'*Evec; % N*N

% E = Wpca*E;
return;