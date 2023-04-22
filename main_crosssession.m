clc, clear, close all;
%% Initialization
% Set params 
param = struct(...
    'fs', 250, ...
    'filter_band', [8, 13], ...
    'fea_type', 'pli', ...  % one of ['psd', 'plv', 'pli', 'wpli', 'corr']
    'n_fea', 12,...   % n_band for psd
    'model',@Model_SVM,... % Choose model: one of ['Model_...', 'Model_SVM']
    'task','20',...    %one of ['all', 'EC', 'EO']
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

param.chan_select = [ones(1,64),0];  %64chs

% Set paths for data   
path_info = 'F:\1.master\1.FinalProject\mycode\Data';
path_data = [path_info, filesep, 'Data'];
% info_enroll = load(fullfile(path_data,'epoch_info.mat'));
info_enroll = Fun_InfoLoader([path_info, filesep, 'Enrollment_Info_',param.task,'.csv']);
info_test = Fun_InfoLoader([path_info, filesep, 'Calibration_Info_',param.task,'.csv']);
disp('Initialization completed');
disp('------------------------------------------------------');

%% Feature extraction of Enrollment
info_fea_enroll = info_enroll;
filetype = 'Enrollment';
path_file = [path_data, filesep, filetype];
path_file_fea_enroll = [path_data, filesep, filetype, '_', param.fea_type, '_', param.task, '_',num2str(param.filter_band(1)), '_', num2str(param.filter_band(2)),'.mat'];
param_fea_enroll = rmfield( param , 'model' ); 

fea_enroll = cell(info_fea_enroll.num, 1);

parfor idx_epoch = 1:info_fea_enroll.num
    mat = load([path_file, filesep, info_fea_enroll.filename{idx_epoch}]);
    data = mat.epoch_data; 
    data = detrend(data(logical(param.chan_select), :));
%     data = detrend(data(param.chan_select, :));
    switch param.filter
        case 'iir'
            data = filtfilt(param.filter_b, param.filter_a, double(data'))';
        case 'fir'
            data = filtfilt(bp,double(data'))';
    end
    fea_enroll{idx_epoch} = Fun_FeaExtract(data, param_fea_enroll);
    disp([info_fea_enroll.filename{idx_epoch}, ' processed!']);
end


fea_enroll = cell2mat(fea_enroll);

% Normalization
if param.fea_type ~= 'psd'
    switch param.nomoralization
        case 'max-min'
            min_fea = min(fea_enroll, [], 2);
            max_fea = max(fea_enroll, [], 2);
            fea_enroll = (fea_enroll - min_fea)./(max_fea - min_fea);
        case 'z-score'
            m_fea=mean(fea_enroll, 2);
            s_fea = std(fea_enroll, 1, 2);
            fea_enroll = bsxfun(@rdivide, bsxfun(@minus, fea_enroll, m_fea), s_fea);
        case 'fisher-z'
            fea_enroll = 0.5*log((1+fea_enroll)./(1-fea_enroll));
        otherwise
            error('Error nomoralization type!');
    end
end

% PCA
% fea_pca_enroll = pca(fea_enroll, 10);   

% 
% path_file_chs=[path_data, filesep, param.fea_type, '_', param.task, '_',num2str(param.filter_band(1)), '_', num2str(param.filter_band(2)), '_chs8.mat'];
% save(path_file_fea, 'fea', 'info_fea', 'param', 'path_data');
% save(path_file_rank, 'ranking');
% path_file_feacom = [path_data, filesep, param.fea_type, '_', param.task, '_',num2str(param.filter_band(1)), '_', num2str(param.filter_band(2)),'_14+14.mat'];
% save(path_file_feacom, 'fea', 'info_fea', 'param', 'path_data', 'fea_ica', 'fea_pca');

%% Feature extraction of Testing
info_fea_test = info_test;
filetype = 'Calibration';
path_file = [path_data, filesep, filetype];
path_file_fea_test = [path_data, filesep, filetype, '_', param.fea_type, '_', param.task, '_',num2str(param.filter_band(1)), '_', num2str(param.filter_band(2)),'.mat'];%_pca10
param_fea_test = rmfield( param , 'model' ); 

fea_test = cell(info_fea_test.num, 1);

parfor idx_epoch = 1:info_fea_test.num
    mat = load([path_file, filesep, info_fea_test.filename{idx_epoch}]);
    data = mat.epoch_data; 
    data = detrend(data(logical(param.chan_select), :));
%     data = detrend(data(param.chan_select, :));
    switch param.filter
        case 'iir'
            data = filtfilt(param.filter_b, param.filter_a, double(data'))';
        case 'fir'
            data = filtfilt(bp,double(data'))';
    end
    fea_test{idx_epoch} = Fun_FeaExtract(data, param_fea_test);
    disp([info_fea_test.filename{idx_epoch}, ' processed!']);
end


fea_test = cell2mat(fea_test);

% Normalization
if param.fea_type ~= 'psd'
    switch param.nomoralization
        case 'max-min'
            min_fea = min(fea_test, [], 2);
            max_fea = max(fea_test, [], 2);
            fea_test = (fea_test - min_fea)./(max_fea - min_fea);
        case 'z-score'
            m_fea=mean(fea_test, 2);
            s_fea = std(fea_test, 1, 2);
            fea_test = bsxfun(@rdivide, bsxfun(@minus, fea_test, m_fea), s_fea);
        case 'fisher-z'
            fea_test = 0.5*log((1+fea_test)./(1-fea_test));
        otherwise
            error('Error nomoralization type!');
    end
end

% PCA
% fea_all = [fea_enroll; fea_test];
% fea_pca_all = my_pca(fea_all, 10);   
% fea_pca_enroll = fea_pca_all([1:size(fea_enroll,1)],:);
% fea_pca_test = fea_pca_all([1:size(fea_test,1)],:);
% 
% save(path_file_fea_test,'fea_enroll', 'fea_test', 'info_fea_enroll','info_fea_test', 'param', 'path_data', 'fea_pca_enroll', 'fea_pca_test');
save(path_file_fea_test,'fea_enroll', 'fea_test', 'info_fea_enroll','info_fea_test', 'param', 'path_data');

%% Training the model
n_enroll = length(unique(info_fea_enroll.id));
fea_train = cell(n_enroll, 1);
id_enroll = cell(n_enroll, 1);
%condition_enroll = cell(n_enroll, 1); 
id_origin = unique(info_fea_enroll.id);
for idx_enroll = 1:n_enroll
    fea_train{idx_enroll} = fea_enroll(info_fea_enroll.id==id_origin(idx_enroll), :);   %fea_pca_enroll
end
disp('------------------------------------------------------');
disp('Training model...');
obj = Fun_ModelEnroll(fea_train);
disp('Enrollment finished!');

%% Crossvalid
threshold = 0;
scorePred = zeros(60, length(obj.model));
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
strout2 = ['the k_fold  accmean of identification model is ',num2str(accmean*100),'% '];
disp(strout2);


%% Testing
info_fea_test.id(isnan(info_fea_test.id)) = 0;
id_true = info_fea_test.id;
n_enroll = obj.n_enroll;
score = zeros(size(fea_test, 1), n_enroll);
for idx_enroll = 1:obj.n_enroll
    [label_row, s] = predict(obj.model{idx_enroll}, fea_test);
    score(:, idx_enroll) = s;
end

[score_result, id_result] = max(score, [], 2); 

id_result(score_result<threshold) = 0;

% Printf result
acc = sum(id_result==id_true)/length(id_true);
disp('------------------------------------------------------');
strout1 = ['the accuracy of identification model is ',num2str(acc*100),'% '];
disp(strout1);



