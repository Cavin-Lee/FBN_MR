%% The main procedure for construct brain networks based different methods
% The original ROI time series of all subjects with labels are stored in a mat file.
% For example, the file name of fMRI data is fMRI.mat where each subject
% corresponds to a cell, and in each cell a matrix is used to store the
% time series in its columns one by one. Also, the file include labels and
% subject index.
% method[1] is PC (Pearson's Correlaton);
% method[2] is SR (Sparse Representation based on Jun et al.'s SLEP);
% method[3] is LR (Low-rank Representation based on Jun et al.'s SLEP)
% method[4] is SLR(Sparse and Low-rank Representation rewritten on Jun et al.'s SLEP)
% method[5] is WGSR
% method[6] is SSGSR
% method[7] is MR

%% basic setting, load data, basic statistics
clear; clc;
root=cd; addpath(genpath([root '/DATA'])); addpath(genpath([root '/FUN']));
%load fMRI80; data=fMRImciNc; clear fMRImciNc;
%load('fmriEMCINC.mat'); data=fmriMCINC(1:137);clear fmriMCINC
load fMRI_njsy; data=fMRImciNc; clear fMRImciNc;

nSubj=length(lab);
nROI=size(data{1},2);
nDegree=size(data{1},1);
method=input('PCC[1],SR[2],LR[3],SLR[4]:');

%% Network learning based on Pearson's correlation (including sparsification with different thresholds)
if method==1
    lambda=[0 10 20 30 40 50 60 70 80 90 99] % the values lies in [0,100] denoting the sparsity degree
    disp('Press any key:'); pause;
    nPar=length(lambda);
    brainNetSet=cell(1,nPar);
    for L=1:nPar
        brainNet=zeros(nROI,nROI,nSubj);
        for i=1:nSubj
            currentNet=corrcoef(data{i});
            currentNet=currentNet-diag(diag(currentNet));% no link to oneself
            threhold=prctile(abs(currentNet(:)),lambda(L)); % fractile quantile
            currentNet(find(abs(currentNet)<=threhold))=0;
            brainNet(:,:,i)=currentNet;
        end
        brainNetSet{L}=brainNet;
        fprintf('Done %d/%d networks!\n',L,nPar);
    end
    save('brainNetSet_PC_njsy.mat','brainNetSet','lab');
end

%% Network learning based on sparse representation(SR) - SLEP
if method==2
    %Parameter setting for SLEP
    ex=-5:5;
    lambda=2.^ex
    disp('Press any key:'); pause;
    nPar=length(lambda);
    brainNetSet=cell(1,nPar);
    
    opts=[];
    opts.init=2;% Starting point: starting from a zero point here
    opts.tFlag=0;% termination criterion
    % abs( funVal(i)- funVal(i-1) ) ¡Ü .tol=10e?4 (default)
    %For the tFlag parameter which has 6 different termination criterion.
    % 0 ? abs( funVal(i)- funVal(i-1) ) ¡Ü .tol.
    % 1 ? abs( funVal(i)- funVal(i-1) ) ¡Ü .tol ¡Á max(funVal(i),1).
    % 2 ? funVal(i) ¡Ü .tol.
    % 3 ? kxi ? xi?1k2 ¡Ü .tol.
    % 4 ? kxi ? xi?1k2 ¡Ü .tol ¡Á max(||xi||_2, 1).
    % 5 ? Run the code for .maxIter iterations.
    opts.nFlag=0;% normalization option: 0-without normalization
    opts.rFlag=0;% regularization % the input parameter 'rho' is a ratio in (0, 1)
    opts.rsL2=0; % the squared two norm term in min  1/2 || A x - y||^2 + 1/2 rsL2 * ||x||_2^2 + z * ||x||_1
    fprintf('\n mFlag=0, lFlag=0 \n');
    opts.mFlag=0;% treating it as compositive function
    opts.lFlag=0;% Nemirovski's line search
    
    for L=1:nPar
        brainNet=zeros(nROI,nROI,nSubj);
        for i=1:nSubj
            tmp=data{i};
            %tmp=tmp-repmat(mean(tmp')',1,nROI);% data centralization
            tmp=tmp-repmat(mean(tmp),nDegree,1);% data centralization
            currentNet=zeros(nROI,nROI);
            for j=1:nROI
                y=[tmp(:,j)];
                A=[tmp(:,setdiff(1:nROI,j))];
                [x, funVal1, ValueL1]= LeastR(A, y, lambda(L), opts);
                currentNet(setdiff(1:nROI,j),j) = x;
            end
            brainNet(:,:,i)=currentNet;
        end
        brainNetSet{L}=brainNet;
        fprintf('Done %d/%d networks!\n',L,nPar);
    end
    save('brainNetSet_SR_njsy.mat','brainNetSet','lab');
end

%% Network learning based on Low-rank Representation - SLEP
if method==3
    ex=-5:5;
    lambda=2.^ex
    disp('Press any key:'); pause;
    nPar=length(lambda);
    brainNetSet=cell(1,nPar);
    opt.epsilon = 10^-5;
    opt.max_itr = 1000;
    for L=1:nPar
        opt=[];
        brainNet=zeros(nROI,nROI,nSubj);
        for i=1:nSubj
            tmp=data{i};
            tmp=tmp-repmat(mean(tmp')',1,nROI);% centrlization
            [currentNet, fval_vec, itr_counter] = accel_grad_mlr(tmp,tmp,lambda(L),opt);
            currentNet=currentNet-diag(diag(currentNet));
            brainNet(:,:,i)=currentNet;
        end
        brainNetSet{L}=brainNet;
        fprintf('Done %d/%d networks!\n',L,nPar);
    end
    save('brainNetSet_LR_njsy.mat','brainNetSet','lab');
end

%% Network learning based on Sparse + Low-rank Representation
if method==4
    ex=-5:5;
    lambda=2.^ex
    z=2.^ex
    disp('Press any key:'); pause;
    nL=length(lambda);
    nZ=length(z);
    brainNetSet=cell(nL,nZ);
    
    opt.epsilon = 10^-5;
    opt.max_itr = 1000;
    for iL=1:nL
        for iZ=1:nZ
            brainNet=zeros(nROI,nROI,nSubj);
            for i=1:nSubj
                tmp=data{i};%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                tmp=tmp-repmat(mean(tmp')',1,nROI);% centrlization
                %parameters for trace + sparse algorithms
                opt.z=z(iZ);
                % learning networks based a revised version of accel_grad_mlr by combining sparse and low-rank regularizers.
                [currentNet, fval_vec, itr_counter] = accel_grad_mlr_qiao(tmp,tmp,lambda(iL),opt);
                currentNet=currentNet-diag(diag(currentNet));
                brainNet(:,:,i)=currentNet;
            end
            brainNetSet{iL,iZ}=brainNet;
            fprintf('Done z=%d,lambda=%d networks!\n',iZ,iL);
        end
    end
    save('brainNetSet_SLR_njsy.mat','brainNetSet','lab');
end
%WGSR
if method==5
    ex=-5:5;
    lambda=2.^ex
    nL=length(lambda);
    for iL=1:nL
        for iZ=1:nL
            for i=1:nSubj
                tmp=data{i};%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                data{i}=tmp-repmat(mean(tmp')',1,nROI);% data centralization
            end
            brainNetSet{iL}= WSGR(data,(iL),lambda(iZ));
            fprintf('lambda=%d ,gamma=%d networks!\n',iL,iZ);
        end
    end
    save('brainNetSet_WSGR_njsy.mat','brainNetSet','lab','-v7.3');
end

%SSGSR
if method==6
    ex=-5:5;
    lambda=2.^ex
    nL=length(lambda);
    for iL=1:nL
        for iZ=1:nL
            for i=1:nSubj
                tmp=data{i};%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                data{i}=tmp-repmat(mean(tmp')',1,nROI);% data centralization
            end
            brainNetSet{iL}= SSGSR(data,(iL),lambda(iZ));
            fprintf('lambda=%d ,gamma=%d networks!\n',iL,iZ);
        end
    end
    save('brainNetSet_SSGSR_njsy.mat','brainNetSet','lab','-v7.3');
end

%MR
if method==7
    ex=-5:5;
    lambda=2.^ex
    nL=length(lambda);
    for iL=1:nL
        for i=1:nSubj
            tmp=data{i};%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            %tmp=tmp-repmat(mean(tmp')',1,nROI);% data centralization
            tmp=tmp-repmat(mean(tmp),nDegree,1);% data centralization
            %parameters for trace + sparse algorithms
            d=1;
            load 264_subNet_14
            % learning networks based a revised version of accel_grad_mlr by combining sparse and low-rank regularizers.
            currentNet = FBNMR(tmp, index,lambda(iL),d);
            currentNet=currentNet-diag(diag(currentNet));
            brainNet(:,:,i)=currentNet;
        end
        brainNetSet{iL}=brainNet;
        fprintf('lambda=%d networks!\n',iL);
    end
    save('brainNetSet_MR_njsy.mat','brainNetSet','lab','-v7.3');
end

%% FBNMR
if method==8
    ex=-5:5;
    lambda=2.^ex
    nL=length(lambda);
    for iL=1:nL
        for i=1:nSubj
            tmp=data{i};%%%%%%%%%%%%%%%%%%%%%%%!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            %tmp=tmp-repmat(mean(tmp')',1,nROI);% data centralization
            tmp=tmp-repmat(mean(tmp),nDegree,1);% data centralization
            %parameters for trace + sparse algorithms
            d=1;
            load 264_subNet_14
            % learning networks based a revised version of accel_grad_mlr by combining sparse and low-rank regularizers.
            currentNet = FBNMR(tmp, index,lambda(iL),d);
            currentNet=currentNet-diag(diag(currentNet));
            brainNet(:,:,i)=currentNet;
        end
        brainNetSet{iL}=brainNet;
        fprintf('lambda=%d networks!\n',iL);
    end
    save('brainNetSet_MR_njsy.mat','brainNetSet','lab','-v7.3');
end