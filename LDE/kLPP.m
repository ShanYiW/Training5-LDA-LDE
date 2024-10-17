function [E] = kLPP(X, k, t, sig2)
% LPP + 高斯核
% X: D*N, D: dimension, N: samples
% K: number of nearest neighbors
% t: para of the heat kernel
% sig2: para of the Gaussian kernel
% E: 
[~,N] = size(X);
X2 = sum(X.*X, 1); % 1*N, [||X_1||2^2 ... ||X_N||2^2 ]
dist = repmat(X2, [N,1]) + repmat(X2', [1,N]) - 2.*X'*X; 
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
Dvec = sum(W, 2); % N*1
% 高斯核
Kernel = exp(-0.5/sig2 .* dist); % N*N  
[Q,Lmd] = eig(Kernel); % 
Lmd = diag(Lmd);
M = diag(ones([N,1])) - Q'*diag(1./sqrt(Dvec))* W * diag(1./sqrt(Dvec))*Q; % N*N
M = (M+M')./2;
[Evec, ~] = eig(M);
E = Q*diag(1./Lmd)*Evec; % N*N
return;

