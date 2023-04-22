clc, clear, close all;
%% Initialization
% Set params 
param = struct(...
    'fs', 250, ...
    'filter_band', [30, 100], ...
    'fea_type', 'plv', ...  % one of ['psd', 'plv', 'pli', 'wpli', 'corr']
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
param.chan_select = [ones(1,64),0];  %64chs
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

% PCA
fea_pca = my_pca(fea, 10);   

% 
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

id_origin = unique(info_fea.id);
for idx_enroll = 1:n_enroll
    fea_enroll{idx_enroll} = fea_pca((info_fea.id==id_origin(idx_enroll))&(info_fea.condition==2), :); 
    fea_test{idx_enroll} = fea_pca((info_fea.id==id_origin(idx_enroll))&(info_fea.condition==1), :); 
    id_enroll{idx_enroll} = repmat([idx_enroll],size(fea_enroll{idx_enroll},1),1);
    id_test{idx_enroll} = repmat([idx_enroll],size(fea_test{idx_enroll},1),1);
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
scorePred = zeros(30, length(obj.model));
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