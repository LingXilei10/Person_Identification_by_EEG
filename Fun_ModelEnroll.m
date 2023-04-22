function obj = Fun_ModelEnroll(fea_enroll)  %fea_other
    % Model enrollment for authentication and identification
    % param:
    %       fea_enroll: the feature of enrollment mat in [n_enroll¡Án_fea]
    %       id_enroll: the id of each epoch
    % return:
    %       obj: svm model for every id
    
    % Import information
    obj.n_enroll = length(fea_enroll);
    obj.model = cell(obj.n_enroll, 1);
    for idx_enroll = 1:obj.n_enroll
        obj.model{idx_enroll} = fitcsvm(fea_enroll{idx_enroll}, ...
            -1*ones(size(fea_enroll{idx_enroll},1), 1), ...  %n¸öepoch=> n¸ö1  -1* 
            'KernelFunction', 'rbf', 'KernelScale', 'auto', ...  
           'Standardize', true, 'OutlierFraction', 0.0008);   % 'gaussian'  rbf  auto   0.0008
    end
end