function [E] = LPP_my(X, k, t)
% X: D*N, D: dimension, N: samples
% K: number of nearest neighbors
% t: para of the heat kernel
% Y: d*N
[~,N] = size(X);
[Wpca] = PCA_DR(X, 0.95); % Wpca: D*r  ORL:0.94
X = Wpca'*X; % D*N -> r*N
%% step 1 构造邻接矩阵
X2 = sum(X.*X, 1); % 1*N, [||X_1||2^2 ... ||X_N||2^2 ]
dist = repmat(X2, [N,1]) + repmat(X2', [1,N]) - 2.*X'*X; 
% dist_{ij}: 样本Xi与Xj的欧氏距离, i,j \in {1,2,..., N}
% k近邻(取并集) 构造邻接矩阵
[~, nei_idx] = sort(dist);
nei_idx = nei_idx(2:k+1, :); % k*N
Adj = false([N,N]);
for j = 1:N
    Adj(nei_idx(:,j), j) = true;
    Adj(j, nei_idx(:,j)) = true;
end
% 构造权重矩阵
W = zeros([N,N]);
W(Adj) = exp(-1.*dist(Adj)./(4*t)); % ||xi-xj||2^2 -> exp( -||xi-xj||2^2/(4t) )
D_vec = sum(W, 2); % N*1
%% 特征值分解
[U,S,V] = svd(X, 'econ'); % X:D*N  U:D*D  S:D*D  V:N*D (D<N)
S = diag(S);
% threshold = sum(S)*0.98;
% chosen=0; cum=0; 
% while cum<threshold
%     chosen = chosen+1;
%     cum = cum + S(chosen);
% end
% S = S(1:chosen); U = U(:,1:chosen); V = V(:,1:chosen);
invS = diag(1./S); % r*r
M = eye(length(S)) - V'*diag(1./sqrt(D_vec))*W*diag(1./sqrt(D_vec))*V; % 机器误差更小
M = (M+M')./2; % (M+M')./2;
[Evec, Eval] = eig(M);
Eval = diag(Eval);
[~,idx_e] = sort(Eval);
Evec = Evec(:,idx_e); % Eval = Eval(idx_e);
E = U*(invS')*Evec;
E = Wpca*E;
return;
