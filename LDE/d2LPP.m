function [L,R] = d2LPP(X, D1,D2, k, t)
% X: D*N  D:维数  N:样本数
% k: 近邻数
% t: 热核函数的参数
% L: D1*D1  R:D2*D2
%% 构造权重矩阵 W \in N*N
N = size(X, 2); % 样本数
X2 = sum(X.*X, 1); % 1*N, [||X_1||2^2 ... ||X_N||2^2 ]
dist = repmat(X2, [N,1]) + repmat(X2', [1,N]) - 2.*X'*X; 
[~, nei_idx] = sort(dist);
nei_idx = nei_idx(2:k+1, :); % k*N
Adj = false([N,N]);
for j = 1:N
    Adj(nei_idx(:,j), j) = true;
    Adj(j, nei_idx(:,j)) = true;
end
W = zeros([N,N]);
W(Adj) = exp(-1.*dist(Adj)./(4*t)); % ||xi-xj||2^2 -> exp( -||xi-xj||2^2/(4t) )
Dvec = sum(W, 2); % N*1
%% 求 L, R
N = size(X, 2);
X3 = zeros([D1, D2, N]); % 每个样本还原成矩阵
for j=1:N
    X3(:,:,j) = reshape(X(:, j), [D1,D2]); % D*1 -> D1*D2, 列优先堆叠
end
L = zeros([D1,D1]); R = zeros([D2,D2]);
converged=false; ratio=Inf; it=1;
while ~converged
    L_old = L;
    % 更新R: D2*D2
    M = zeros([D2,D2]);
    C = zeros([D2,D2]); 
    for j=1:N
        for i=1:N
            Lt_Xi_Xj = L'*(X3(:,:,i)-X3(:,:,j)); % D1*D2
            M = M + Lt_Xi_Xj'*Lt_Xi_Xj .* W(i,j);
        end
        Xjt_L = X3(:,:,j)'*L; % D2*D1
        C = C + Xjt_L*Xjt_L' .* Dvec(j);
    end
    [R, ~] = eig(M, C); % D2*D2  Eval = diag(Eval);
    % 更新L: D1*D1
    M = zeros([D1,D1]); 
    C = zeros([D1,D1]); 
    for j=1:N
        for i=1:N
            Xi_Xj_R = (X3(:,:,i)-X3(:,:,j))*R; % D1*D2
            M = M + Xi_Xj_R*Xi_Xj_R' .* W(i,j); % D1*D1
        end
        Xj_R = X3(:,:,j)*R; % D1*D2
        C = C + Xj_R*Xj_R' .* Dvec(j); % D1*D1
    end
    [L, ~] = eig(M, C); % D1*D1
    % 判敛
    LLp_fro2 = sum(sum((L_old-L).*(L_old-L)));
    L_fro2 = sum(sum(L.*L));
    ratio_old = ratio;
    ratio = LLp_fro2/L_fro2;
    if abs(ratio-ratio_old) < 1e-2 || it>30; converged=true; end % 
    it = it+1;
end
% figure; plot(ratio_arr);
return;