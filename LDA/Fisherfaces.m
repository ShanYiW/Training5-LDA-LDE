function [Wfld] = Fisherfaces(X, gnd)
% X: D*N
% gnd: N*1
% Wpca: D*N  Wfld: N*c(min version) or N*N(max version)
[D,N] = size(X);
class = unique(gnd); % 类别标签
C = length(class); % 类别数
mX = mean(X,2); % D*1  全局均值

Xo = X - repmat(mX, [1,N]); % D*N
[Wpca,~,~] = svd(Xo); % U:D*D  S:D*N  V:N*N
Wpca = Wpca(:,1:N);
X = Wpca'*X; % D*N -> N*N
mX = mean(X,2); % N*1  全局均值
D = size(X,1);

Nc  = zeros([C, 1]); % 各类的样本数
Xsort_o = zeros([D,N]); % 按类分开的, 每类减去类内均值的数据矩阵
Mu = zeros([D,C]);
cursor = 0;
for c=1:C
    idx_c = gnd==class(c);
    Nc(c) = length(gnd(idx_c));
    Xc = X(:,idx_c); % D*Nc 该类的样本
    mu = mean(Xc, 2);
    Mu(:,c) = mu - mX;
%     Mu(:,c) = sqrt(Nc(c)).*(mu - mX); % 
    Xsort_o(:,cursor+1:cursor+Nc(c)) = Xc - repmat(mu, [1,Nc(c)]);
    cursor = cursor+Nc(c);
end
% Xsort_o2 = zeros([D,N]);
% for j=1:N
%     Xsort_o2(:,j) = X(:,j) - Mu(:,gnd(j)) - mX;
% end
%% Fisherfaces
% Xo = X - repmat(mX, [1,N]); % D*N  rank(Xo) = N-1
% [Wpca,~,~] = svd(Xo); % U:D*D  S:D*N  V:N*N
% Wpca = Wpca(:,1:N); % D*D -> D*N
% Sb_half = Wpca'*Mu; % (N*D)*(D*c)=N*c  rank = c-1
% Sw_half = Wpca'*Xsort_o; % (N*D)*(D*N)=N*N  rank = N-c
Sb_half = Mu; 
Sw_half = Xsort_o; 
%% max Tr(W'Wpca' Sb Wpca W), s.t. W'Wpca' Sw Wpca W = I
[U,S,~] = svd(Sw_half); % U,S:N*N
[row,col] = size(S);
S = diag(S);
if row >= col % S:col*1
    invS = [diag(1./S), zeros([col, row-col])]; % col*row
else % row < col
    invS = [diag(1./S); zeros([col-row, row])]; % col*row
end
Core_half = invS*U'*Sb_half; % (N*N)*(N*N)*(N*D)*(D*c)=N*c
% Core = Core_half*Core_half'; % (N*c)*(c*N) = N*N
% [Evec, ~] = eig(Core); % Rank(Core) = c-1 << N  Eval = diag(Eval);
% Evec = Evec(:,end:-1:1); % N*N  最大化
[Evec,~,~] = svd(Core_half); % s = diag(s);
Wfld = U*invS'*Evec; % N*N
Wfld = Wpca*Wfld; % (D*N)*(N*N) = D*N
%% min Tr(W'Wpca' Sw Wpca W), s.t. W'Wpca' Sb Wpca W = I
% [U,S,~] = svd(Sb_half); % U:N*N  S:N*c
% S = diag(S); % N*c -> c*1
% invS = [diag(1./S), zeros([C, N-C])]; % c*N
% Core_half = invS*U'*Sw_half; % (c*N)*(N*N)*(N*D)*(D*N)=c*N
% % Core = Core_half*Core_half'; % (c*N)*(N*c) = c*c
% % [Evec, ~] = eig(Core); % Evec: c*c  Rank(Core) = c-1 << N  Eval=diag(Eval);
% [Evec, ~,~] = svd(Core_half); % Evec: c*c
% Evec = Evec(:,end:-1:1); % 最小化
% Wfld = U*invS'*Evec; % (N*N)*(N*c)*(c*c) = N*c
return;