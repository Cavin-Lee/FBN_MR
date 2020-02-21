function []=save_optimal_network(meth_Net,BrainNet,label,result_dir,varargin)
switch meth_Net
    case {'PC','tHOFC','aHOFC'}
        opt_BrainNet=BrainNet;
    case {'SR','SLR','SGR','GSR','WSR','WSGR','SSGSR','dHOFC'}
        Acc_para=varargin{1};
        [~,B]=max(Acc_para);
        opt_BrainNet=BrainNet{B(1)};
end

label_negative=find(label==-1);
label_positive=find(label==1);
BrainNet_negative_mean=mean(opt_BrainNet(:,:,label_negative),3);
BrainNet_positive_mean=mean(opt_BrainNet(:,:,label_positive),3);

%figure;
figure('visible','off');
subplot(1,2,1);
imagesc(BrainNet_negative_mean);
colormap jet
colorbar
axis square
xlabel('ROI');
ylabel('ROI');
title('label = -1');

subplot(1,2,2);
imagesc(BrainNet_positive_mean);
colormap jet
colorbar
axis square
xlabel('ROI');
ylabel('ROI');
title('label = 1');
print(gcf,'-r1000','-dtiff',char(strcat(result_dir,'/Mean_optimal_network.tiff')));
save (char(strcat(result_dir,'/Mean_optimal_negativeLabel_network.mat')),'BrainNet_negative_mean');
save (char(strcat(result_dir,'/Mean_optimal_positiveLabel_network.mat')),'BrainNet_positive_mean');
