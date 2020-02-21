function W = SelfpaceLeastR(data,lambda,gamma)

%% Function SelfpaceLeastR
%      Least Squares Loss with the L1-norm Regularization and selfpace
% under the SLEP package
%% Problem
%
%  min  1/2 v|| data - data * W||^2 + lambda * ||W||_1-gamma*||v||_1
%  W (ii) ~= 0
%

%
%% Input parameters:
%
%  data-         Matrix of size m x n
%                data can be a dense matrix
%                         a sparse matrix
%                         or a DCT matrix
%  label -        Response vector (of size mx1)
%  lambda -        L_1 norm regularization parameter (lambda >=0)
%  gamma-      selfpace regularization parameter (gamma >=0, )
%
%% Output parameters:
%  W-         Solution
%
%% Copyright WeikaiLi@cqjtu  email :leeweikai@outlook.com
%
[sample,nROI]=size(data);

v=ones(1,sample);
J_old = rand(1,1);
J_new = rand(1,1);
maxIter = 100;
Iter=1;
epsilon=10^-4;
W_tmp=zeros(nROI,nROI);

%%         SPL
while abs((J_old-J_new)/J_old)>epsilon %end condition
    %W=W_tmp;
    J_old=J_new;
    opts=[];
    opts.init=2;% Starting point: starting from a zero point here
    opts.tFlag=0;% termination criterion
    opts.nFlag=0;% normalization option: 0-without normalization
    opts.rFlag=0;% regularization % the input parameter 'rho' is a ratio in (0, 1)
    opts.rsL2=0; % the squared two norm term in min	 1/2 || A x - y||^2 + 1/2 rsL2 * ||x||_2^2 + z * ||x||_1
    %fprintf('\n mFlag=0, lFlag=0 \n');
    opts.mFlag=0;% treating it as compositive function
    opts.lFlag=0;% Nemirovski's line search
    
    
    tmp_old=diag(v)*data;
    index=find(v==0);
    tmp=tmp_old(setdiff(1:sample,index),:);% this place used the hard threshold and filter out the noise sample
    tmp=tmp-repmat(mean(tmp')',1,nROI);% data centralization
    currentNet=zeros(nROI,nROI);
    for j=1:nROI
        y=[tmp(:,j)];
        A=[tmp(:,setdiff(1:nROI,j))];
        [x, funVal1, ValueL1]= LeastR(A, y, lambda, opts);
        currentNet(setdiff(1:nROI,j),j) = x;
    end
    brainNet=currentNet;
    W_tmp=brainNet;
    S_val=sum(sum(abs(W_tmp)));
    Q_val=(sum(abs(v)));
    J_new = diag(v)*norm(data-data*W_tmp,'fro')^2+lambda*S_val-gamma*Q_val;
    
    error=abs(sum(((data-data*W_tmp))'))';
  
    [d c]=kmeans(error,2);
   
    ind=find(c==max(c));
    
    index= find(d==ind);
    num=length(index);
    sorterror=sort(error(index));
    threshold=sorterror(ceil(num*gamma));
    v=v.*(error'<threshold);%updata V
    
%     sorterror=sort(error);
%     threshold=sorterror(ceil(sample*gamma));
%     v=v.*(error<threshold);%updata V
    %     gamma=gamma*1.1;
    %     lambda=lambda*1.1;%updata gamma and lambda
    Iter=Iter+1;
    
    if Iter>maxIter
        fprintf('.\n')
        break;
    end
end
W=W_tmp;% accyracy  0.8462

%%    Kmeans for filter noise
% %W=W_tmp;
% J_old=J_new;
% opts=[];
% opts.init=2;% Starting point: starting from a zero point here
% opts.tFlag=0;% termination criterion
% opts.nFlag=0;% normalization option: 0-without normalization
% opts.rFlag=0;% regularization % the input parameter 'rho' is a ratio in (0, 1)
% opts.rsL2=0; % the squared two norm term in min	 1/2 || A x - y||^2 + 1/2 rsL2 * ||x||_2^2 + z * ||x||_1
% %fprintf('\n mFlag=0, lFlag=0 \n');
% opts.mFlag=0;% treating it as compositive function
% opts.lFlag=0;% Nemirovski's line search
%
%
% tmp_old=diag(v)*data;
% index=find(v==0);
% tmp=tmp_old(setdiff(1:sample,index),:);% this place used the hard threshold and filter out the noise sample
% tmp=tmp-repmat(mean(tmp')',1,nROI);% data centralization
% currentNet=zeros(nROI,nROI);
% for j=1:nROI
%     y=[tmp(:,j)];
%     A=[tmp(:,setdiff(1:nROI,j))];
%     [x, funVal1, ValueL1]= LeastR(A, y, lambda, opts);
%     currentNet(setdiff(1:nROI,j),j) = x;
% end
% brainNet=currentNet;
% W_tmp=brainNet;
% error=abs(sum((diag(v)*(data-data*W_tmp))'));
% [d c]=kmeans(error,2);
% ind=find(c==max(c));
% index= find(d==ind);
% v(index)=0;
% tmp_old=diag(v)*data;
% index=find(v==0);
% tmp=tmp_old(setdiff(1:sample,index),:);% this place used the hard threshold and filter out the noise sample
% tmp=tmp-repmat(mean(tmp')',1,nROI);% data centralization
% currentNet=zeros(nROI,nROI);
% for j=1:nROI
%     y=[tmp(:,j)];
%     A=[tmp(:,setdiff(1:nROI,j))];
%     [x, funVal1, ValueL1]= LeastR(A, y, lambda, opts);
%     currentNet(setdiff(1:nROI,j),j) = x;
% end
% brainNet=currentNet;
%
% W=brainNet;
%% accuracy =79.12
