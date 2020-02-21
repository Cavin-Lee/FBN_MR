function W=PSOfirst(A,L)
%% 清空环境


%% 参数设置
w=0.9;%权值 将影响PSO 的全局与局部搜优能力， 值较大，全局搜优能力强，局部搜优能力弱;反之，则局部搜优能力增强，而全局搜优能力减弱。
c1=0.1;%加速度，影响收敛速度
c2=0.1;
nROI=size(A,2);
dim=nROI*nROI;
swarmsize=1;%粒子群规模，表示有100个解的空间
maxiter=200;%最大循环次数，影响时间
minfit=0.001;%最小适应值
vmax=0.01;
vmin=-0.01;
ub=ones(1,dim);%解向量的最大限制
lb=-ones(1,dim);%解向量的最小限制

%% 种群初始化

swarm=rand(swarmsize,dim)*2-1;%粒子群位置矩阵
vstep=rand(swarmsize,dim)*(vmax-vmin)+vmin;%粒子群速度矩阵
fswarm=zeros(swarmsize,1);%预设空矩阵，存放适应值
for i=1:swarmsize
    X=swarm(i,:);
   
     fswarm(i)=fitness(X,A,L);
    %fswarm(i,:)=feval(jn,swarm(i,:));%以粒子群位置的第i行为输入，求函数值，对应输出给适应值
end

%% 个体极值和群体极值
[bestf,bestindex]=min(fswarm);%求得适应值中的最小适应值，和，其所在的序列
gbest=swarm;%暂时的个体最优解为自己
fgbest=fswarm;%暂时的个体最优适应值
zbest=swarm(bestindex,:);%所在序列的对应的解矩阵序列，全局最佳解
fzbest=bestf;%全局最优适应值


%% 迭代寻优
iter=0;



while((iter<maxiter)&&(fzbest>minfit))
    for j=1:swarmsize
        % 速度更新
        vstep(j,:)=w*vstep(j,:)+c1*rand*(gbest(j,:)-swarm(j,:))+c2*rand*(zbest-swarm(j,:));
        if vstep(j,:)>vmax  
            vstep(j,:)=vmax;%速度限制
        end
        if vstep(j,:)<vmin
            vstep(j,:)=vmin;
        end
        % 位置更新
        swarm(j,:)=swarm(j,:)+vstep(j,:);
        for k=1:dim
            if swarm(j,k)>ub(k)
                swarm(j,k)=ub(k);%位置限制
            end
            if swarm(j,k)<lb(k)
                swarm(j,k)=lb(k);
            end
        end
       
        % 适应值        
         X=swarm(j,:);
       
         fswarm(j,:)=fitness(X,A,L);
        % 可在此处增加约束条件，若满足约束条件，则进行适应值计算
        
        %
        % 个体最优更新
        if fswarm(j)>fgbest(j) %如果当前的函数值比个体最优值小
            gbest(j,:)=swarm(j,:);%个体最优解更新
            fgbest(j)=fswarm(j);%个体最优值更新
        end
        % 群体最优更新
        if fswarm(j)>fzbest%如果当前的函数值比群体最优值大
            zbest=swarm(j,:);%群体最优解更新
            fzbest=fswarm(j);%群体最优值更新
        end
    end
    iter=iter+1;
   

 
 
end

W=reshape(zbest,nROI,nROI);
return
function z=fitness(X,A,L)
Wp=reshape(X,size(A,2),size(A,2));
z=0.5*norm(Wp-A'*A,'fro')^2+L*sum(sum(abs(Wp)));

return

