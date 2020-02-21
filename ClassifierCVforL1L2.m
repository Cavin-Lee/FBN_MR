%% Classification based on constructed brain networks
%  Select network parameter L1 and L2 through CV, based on
%    1) t-test with p=0.01 for selecting features
%    2) linear SVM with default C=1 for classification
% Written by Lishan Qiao @UNC, 11/09/2016
%%
clear
close all
p_value=0.01; %p-value of t-test for feature selection
dsvname='brainNetSet_MR_test_njsy.mat';
load brainNetSet_MR_test_njsy.mat; %Brain network sets based on different net parameters.
root=cd; addpath(genpath([root '/DATA'])); addpath(genpath([root '/FUN']));

nROI=size(brainNetSet{1,1},1); %ROI size
nSubj=length(lab); %# of subject
[nL nZ]=size(brainNetSet); % # of candidate parameter Lambda1 and Lambda2


kfoldout=nSubj; % LOO CV for outer loop
kfoldin=nSubj-1; % inner LOO CV

c_out = cvpartition(nSubj,'k',kfoldout); % for outer CV

Test_res = zeros(1,kfoldout); % for final test results
for fdout=1:c_out.NumTestSets
    %% CV
    Vali_res=zeros(nL,nZ); % for results on validation set (split from training set)
    for iL=1:nL % corresponds to regularized parameter lambda 2 in our model
        for iZ=1:nZ % lambda 1 (L1-norm regularized parameter in our model)
            data=zeros(nROI^2,nSubj);
            for i=1:nSubj
                originalNet = brainNetSet{iL,iZ}(:,:,i);
                originalNet=originalNet-diag(diag(originalNet)); % remove the non-informative diagal elements
                originalNet=(originalNet+originalNet')/2; % symmetrization
                originalNet=triu(originalNet);
                data(:,i)=reshape(originalNet,nROI^2,1);
            end
            idx=find(sum(abs(data'))>1e-12); % for removing the trivial zero-rows
            data=data(idx,:);
            
            Train_dat = data(:,training(c_out,fdout));
            Train_lab = lab(training(c_out,fdout));
            Test_dat = data(:,test(c_out,fdout));
            Test_lab = lab(test(c_out,fdout));
            
            c_in = cvpartition(length(Train_lab),'k',kfoldin); % leave one out for validation (i.e., parameter slection)
            acc=0; % for save current validation result
            for fdin=1:c_in.NumTestSets
                
                % partion the training samples into a training subset and a
                % validation subset for selecing optimal regularized network parameters.
                InTrain_dat = Train_dat(:,training(c_in,fdin));
                InTrain_lab = Train_lab(training(c_in,fdin));
                Vali_dat = Train_dat(:,test(c_in,fdin));
                Vali_lab = Train_lab(test(c_in,fdin));
                
                POSITIVE_data = InTrain_dat(:,InTrain_lab==1);
                NEGATIVE_data = InTrain_dat(:,InTrain_lab==-1);
                
                % t-test for feature selection
                [tad,p_tmp] = ttest2(double(POSITIVE_data'), double(NEGATIVE_data'));
                [ordered_p,indp]=sort(p_tmp);
                index = indp(ordered_p<p_value);
                
                InTrain_dat = InTrain_dat(index,:);
                Vali_dat = Vali_dat(index,:);

                % scaling the data, which is recommended by LIBSVM toolbox for classification.
                MaxV=(max(InTrain_dat'))';
                MinV=(min(InTrain_dat'))';
                [R,C]= size(InTrain_dat);
                InTrain_dat=2*(InTrain_dat-repmat(MinV,1,C))./(repmat(MaxV,1,C)-repmat(MinV,1,C))-1;
                [R,C]= size(Vali_dat);
                Vali_dat=2*(Vali_dat-repmat(MinV,1,C))./(repmat(MaxV,1,C)-repmat(MinV,1,C))-1;
                
                cmd = ['-t 0 -c 1 -q'];% linear kernel
                model = svmtrain(InTrain_lab', InTrain_dat', cmd); % Linear Kernel
                
                [predict_label, accuracy_svm, prob_estimates] = svmpredict(Vali_lab', Vali_dat', model, '-q');
                
                acc = acc+sum(predict_label==Vali_lab')/length(Vali_lab);
                fprintf('current foldin=%d, Z=%d, Lambda=%d, current foldout=%d\n',fdin,iZ,iL,fdout);
            end
            Vali_res(iL,iZ)=acc/c_in.NumTestSets;
        end
    end
    
    [row col]=find(Vali_res==max(Vali_res(:)));% find optimal regularized network parameters
    lMax=row(end); zMax=col(end);

    %% Test based on the optimal parameters obtained from the training data.
    data=zeros(nROI^2,nSubj);
    for i=1:nSubj
        originalNet = brainNetSet{lMax,zMax}(:,:,i);
        originalNet=originalNet-diag(diag(originalNet));
        originalNet=(originalNet+originalNet')/2;
        originalNet=triu(originalNet);
        data(:,i)=reshape(originalNet,nROI^2,1);
    end
    idx=find(sum(abs(data'))>1e-12);
    data=data(idx,:);
    
    Train_dat = data(:,training(c_out,fdout));
    Train_lab = lab(training(c_out,fdout));
    Test_dat = data(:,test(c_out,fdout));
    Test_lab = lab(test(c_out,fdout));
    
    POSITIVE_data = Train_dat(:,Train_lab==1);
    NEGATIVE_data = Train_dat(:,Train_lab==-1);
    
    % t-test for feature selection
    [tad,p_tmp] = ttest2(double(POSITIVE_data'), double(NEGATIVE_data'));
    [ordered_p,indp]=sort(p_tmp);
    index = indp(ordered_p<p_value);
    Train_dat = Train_dat(index,:);
    Test_dat = Test_dat(index,:);
    
    % scaling the data
    MaxV=(max(Train_dat'))';
    MinV=(min(Train_dat'))';
    [R,C]= size(Train_dat);
    Train_dat=2*(Train_dat-repmat(MinV,1,C))./(repmat(MaxV,1,C)-repmat(MinV,1,C))-1;
    [R,C]= size(Test_dat);
    Test_dat=2*(Test_dat-repmat(MinV,1,C))./(repmat(MaxV,1,C)-repmat(MinV,1,C))-1;
    
    cmd = ['-t 0 -c 1 -q'];% linear kernel
    model = svmtrain(Train_lab', Train_dat', cmd); % Linear Kernel
    
    [predict_label, accuracy_svm, prob_estimates] = svmpredict(Test_lab', Test_dat', model, '-q');
    Test_estimates(test(c_out,fdout))=prob_estimates;
    Test_label(test(c_out,fdout))=predict_label;
    Test_res(test(c_out,fdout)) = sum(predict_label==Test_lab')/length(Test_lab);%accuracy;
    fprintf('current fold=%d.\n',fdout);
end
save(dsvname,'Test_estimates','Test_label','Test_res')