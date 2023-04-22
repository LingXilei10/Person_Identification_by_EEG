clc, clear, close all;
%% Initialization
% Set params 
param = struct(...
    'fs', 250, ...
    'filter_band', [30, 100], ...
    'fea_type', 'pli', ...  % one of ['psd', 'plv', 'pli', 'wpli', 'corr']
    'n_fea', 12,...   % n_band for psd
    'model',@Model_SVM,... % Choose model: one of ['Model_...', 'Model_SVM']
    'task','all',...    %one of ['all', 'EC', 'EO']
    'nomoralization','z-score',...  %one of ['max-min', 'z-score', 'fisher-z']
    'filter','fir'...   %one of ['fir', 'iir']
    );

% Filter selection
switch param.filter
    case 'iir'
        [param.filter_b, param.filter_a] = butter(4, param.filter_band./(param.fs/2));  
    case 'fir'
        bp = designfilt('bandpassfir', 'FilterOrder', 100, 'CutoffFrequency1', param.filter_band(1),...
            'CutoffFrequency2', param.filter_band(2), 'StopbandAttenuation1', 60, 'PassbandRipple', 1,...
            'StopbandAttenuation2', 60, 'SampleRate', param.fs, 'DesignMethod', 'cls', 'ZeroPhase', true);
    otherwise
        error('Error filter type!');
end

% Channel selection  
% param.chan_select = [ones(1, 8), zeros(1, 57)];     % n
% ch = c(randperm(numel(c))); %  random
param.chan_select = [ones(1,64),0];  %64chs

% n_chs = sum(param.chan_select);
% param.chan_select = [28,33,1,60,2,17,52,3];   % relieff
% Set paths for data   
path_info = 'F:\1.master\1.FinalProject\mycode\Data';
path_data = [path_info, filesep, 'Data'];
% info_enroll = load(fullfile(path_data,'epoch_info.mat'));
info_enroll = Fun_InfoLoader([path_info, filesep, 'Enrollment_Info_',param.task,'.csv']);
disp('Initialization completed');
disp('------------------------------------------------------');

%% Feature extraction
info_fea = info_enroll;
filetype = 'Enrollment';
path_file = [path_data, filesep, filetype];
path_file_fea = [path_data, filesep, param.fea_type, '_', param.task, '_',num2str(param.filter_band(1)), '_', num2str(param.filter_band(2)),'.mat'];
param_fea = rmfield( param , 'model' ); 

fea = cell(info_fea.num, 1);

parfor idx_epoch = 1:info_fea.num
    mat = load([path_file, filesep, info_fea.filename{idx_epoch}]);
    data = mat.epoch_data; 
    data = detrend(data(logical(param.chan_select), :));
%     data = detrend(data(param.chan_select, :));
    switch param.filter
        case 'iir'
            data = filtfilt(param.filter_b, param.filter_a, double(data'))';
        case 'fir'
            data = filtfilt(bp,double(data'))';
    end
    fea{idx_epoch} = Fun_FeaExtract(data, param_fea);
    disp([info_fea.filename{idx_epoch}, ' processed!']);
end

% Plot
% figure(1); plot(data);
% figure(2);plot(data(1,:));
% figure(3);imagesc(data);colormap(gca,jet);colorbar; axis xy;
% xlabel('Sample');ylabel('Channel');title('epoch000046');  %title('epoch000046 after preprocess');

fea = cell2mat(fea);

% Normalization
if param.fea_type ~= 'psd'
    switch param.nomoralization
        case 'max-min'
            min_fea = min(fea, [], 2);
            max_fea = max(fea, [], 2);
            fea = (fea - min_fea)./(max_fea - min_fea);
        case 'z-score'
            m_fea=mean(fea, 2);
            s_fea = std(fea, 1, 2);
            fea = bsxfun(@rdivide, bsxfun(@minus, fea, m_fea), s_fea);
        case 'fisher-z'
            fea = 0.5*log((1+fea)./(1-fea));
        otherwise
            error('Error nomoralization type!');
    end
end

%% channel pairs importance ranking
% relieff
% coordinate = [];
% integral = zeros(n_chs, 1);
% for idx_i = 1:n_chs
%     for idx_j = idx_i+1:n_chs
%         coordinate = [coordinate; idx_i, idx_j];
%     end
% end
% 
% idd = repmat([1:95]',1,60)';
% idd = idd(:);

% [ranking,~] = relieff(fea,idd, 60);
% 
% for idx = 1:size(ranking,2)
%     coordinate(ranking(1,idx),3) = idx;  
% end
% 
% for idx= 1:size(ranking,2)
%     integral(coordinate(idx, 1),1) = integral(coordinate(idx, 1),1)+ coordinate(idx, 3);
%     integral(coordinate(idx, 2),1) = integral(coordinate(idx, 2),1)+ coordinate(idx, 3);
% end
% [integral_result, chs_result] = sort(integral, 'ascend');
% param.chan_select=chs_result(1:8,1);
% fea = fea(:, ranking(1, 1:28));

% opts = statset('Display','iter');   % 显示迭代过程
% [ranking,wights] = relieff(fea_enroll{1}, -1*ones(size(fea_enroll{1},1), 1), 10, 'method', 'svr', 'model', obj.model{1}, 'options', opts);


% ICA
recon_ica = rica(fea, 14);
fea_ica = transform(recon_ica, fea);
% 28 85.5789% 16.6737%  20 86.6316% 13.8526%
% 18 88.1053% 13.0526%  16 84.7368% 14.4%  
% 15 87.7895% 12.1684%  14 88.4211% 10.9053% 
% 13 86.8421% 10.6947%  12 86.5263% 10.0842% 
% 11 85.3684%  9.7684%  10 84.8421%  9.8316%

% FastICA
[icasig, A, W] = fastica(fea', 'lastEig', 90);

fea_fastica = icasig';

% Sparsefilt
% sparse_matrix = sparsefilt(fea, 28);
% fea = transform(sparse_matrix, fea);   % 26.7368%   8.0842% 

% PCA
m_fea_fastica=mean(fea_fastica, 2);
s_fea_fastica = std(fea_fastica, 1, 2);
fea_fasticaz = bsxfun(@rdivide, bsxfun(@minus, fea_fastica, m_fea_fastica), s_fea_fastica);
  
[coeff, score, latent] = pca(fea_fastica);
variance_explained = latent/sum(latent);

pareto(latent);
xlabel('Principal Component');
ylabel('Variance Explained (%)');

sumpor=zeros(size(variance_explained,1),1);
for idx = 1:size(variance_explained,1)
    sumpor = sum( variance_explained([1:idx],1))/sum(variance_explained);
end


% fea_pca = my_pca(fea_fastica, 16); 
%  28 83.0526%  18.1263%  20 87.5789%   14.2105%   
%  18 88.3158%  13.6%     16 87.4737    13.3053% 
%  15 89.4737%  12.4842%  14 88.8421%/89.7895%   12.0632%  
%  13 88.3158%  11.4947%  12 89.8947%   10.2316%
%  11 88%        9.6421%  10 89.4737%/90.7368%    9.2632%/9.5579% 
%   9 87.0526%   8.6316%   8 86.2105%    8.0842% 
%   7 80.4211%   8.4632%            

% tsne
fea_pca = my_pca(fea, 50); 
fea_tsne = tsne(fea_pca);

% 
% %  Fast-NCA
% net = fastnca(fea', 28);
% fea = net(fea');

% % L1
% idd = repmat([1:95]',1,60)';
% idd = idd(:);
% [beta, FitInfo] = lasso(fea, idd, 'Lambda', 0.1);
% 
% % 可视化结果
% lassoPlot(beta, FitInfo, 'PlotType', 'Lambda', 'XScale', 'log');
% legend('setosa','versicolor','virginica','Location','best');

% path_file_chs=[path_data, filesep, param.fea_type, '_', param.task, '_',num2str(param.filter_band(1)), '_', num2str(param.filter_band(2)), '_chs8.mat'];
% save(path_file_fea, 'fea', 'info_fea', 'param', 'path_data');
% save(path_file_rank, 'ranking');
% path_file_feacom = [path_data, filesep, param.fea_type, '_', param.task, '_',num2str(param.filter_band(1)), '_', num2str(param.filter_band(2)),'_14+14.mat'];
% save(path_file_feacom, 'fea', 'info_fea', 'param', 'path_data', 'fea_ica', 'fea_pca');
%% Training the model
n_enroll = length(unique(info_fea.id));
fea_enroll = cell(n_enroll, 1);
fea_test = cell(n_enroll, 1); 
fea_sort = cell(n_enroll, 1); 
id_sort = cell(n_enroll, 1); 
id_enroll = cell(n_enroll, 1);
id_test = cell(n_enroll, 1);
%condition_enroll = cell(n_enroll, 1); 

div = sum(info_fea.id==info_fea.id(1))/5;   %  60epoch=> 12div 5epoch  
a = ones(1, 4*div);  %a = ones(1, 5*div);  b = zeros(1, div);
b = zeros(1, 1*div);
c = [a, b];  
%c = c(randperm(numel(c))); 
%enroll_choose = repmat(c, [n_enroll 1]);
%test_choose = ~enroll_n;
id_origin = unique(info_fea.id);
% fea_com = [fea_fastica, fea_pca]; %[fea_ica(:,[1:7]), fea_pca(:,[1:7])];
for idx_enroll = 1:n_enroll
    fea_sort{idx_enroll} = fea_tsne(info_fea.id==id_origin(idx_enroll), :); 
    fea_enroll{idx_enroll} = fea_sort{idx_enroll}(c==1, :);
    fea_test{idx_enroll} = fea_sort{idx_enroll}(c==0, :);
    id_sort{idx_enroll} = info_fea.id(info_fea.id==id_origin(idx_enroll));
    id_enroll{idx_enroll} = id_sort{idx_enroll}(c==1, :);
    %id_test{idx_enroll} = id_sort{idx_enroll}(c==0, :);
    id_test{idx_enroll} = repmat([idx_enroll],1*div,1);
end
disp('------------------------------------------------------');
disp('Training model...');
obj = Fun_ModelEnroll(fea_enroll);
disp('Enrollment finished!');


%% Testing
id_true = cell2mat(id_test);
fea_test = cell2mat(fea_test);
n_enroll = obj.n_enroll;
score = zeros(size(fea_test, 1), n_enroll);
for idx_enroll = 1:obj.n_enroll
    [label_row, s] = predict(obj.model{idx_enroll}, fea_test);
    score(:, idx_enroll) = s;
end

[score_result, id_result] = max(score, [], 2); 
% threshold = min(score_result);
threshold = 0;
id_result(score_result<threshold) = 0;

% Printf result
acc = sum(id_result==id_true)/length(id_true);
disp('------------------------------------------------------');
strout1 = ['the accuracy of identification model is ',num2str(acc*100),'% '];
disp(strout1);



%% Crossvalid
scorePred = zeros(length(a), length(obj.model));
outlierRate = zeros(length(obj.model), 1);
for idx_cross = 1: length(obj.model)
    CVSVMModel = crossval(obj.model{idx_cross});
    [~,scorePred(:,idx_cross)] = kfoldPredict(CVSVMModel);
    outlierRate(idx_cross,1) = mean(scorePred(:,idx_cross)<threshold);  % each model
    accRate(idx_cross,1) = mean(scorePred(:,idx_cross)>threshold);
end
generalization = mean(outlierRate);
accmean = mean(accRate);
disp('------------------------------------------------------');
strout2 = ['the genneralization of identification model is ',num2str(generalization*100),'% '];
disp(strout2);

% 
% % one class svm use libsvm
% % label=ones(Number Of your training instances,1); % You should generate labels for your only class!
% % model = svmtrain( label, Training Data , '-s 2 -t 2  -n 0.5' ) ; % You can change the parameters  '-s 2 -t 2 -g 0.00001 -n 0.01'
% % [predicted_label,accuracy]=svmpredict(TestLabels,Test Set, model);
% %