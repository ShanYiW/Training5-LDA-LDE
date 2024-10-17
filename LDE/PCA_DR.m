function [Evec] = PCA_DR(X, ratio)
% X: D*N  D:特征数  N:样本数
% ratio: [0,1]
% Evec: D*r  r<D
N = size(X,2); % 样本数
mX = mean(X,2); % D*1
X = X - repmat(mX, [1,N]);

XXt = X*(X'); % D*D
[Evec, Eval] = eig(XXt); % Evec, Eval:D*D 
Eval = diag(Eval); % D*1
[~, idx_e] = sort(-Eval); % 降序排序
Eval = Eval(idx_e);
Evec = Evec(:,idx_e);

if ratio<1 && ratio>0
    threshold=sum(Eval)*ratio;
    chosen=0; cum=0; 
    while cum<threshold
        chosen = chosen + 1;
        cum = cum + Eval(chosen);
    end
    % Eval = Eval(1:chosen); % chosen*1
    Evec = Evec(:,1:chosen); % D*chosen
elseif ratio>1 % 保留前ratio个奇异值
    Evec = Evec(:,1:ratio); 
end
return;