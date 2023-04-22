function [info] = Fun_InfoLoader(path_info)
    % Load information of dataset from the given path.
    % param:
    %       path_info: the path of information file (.csv)
    % return:
    %       info: struct information (filename, id, session, condition, usage, num)
    
    % Import information
    info_raw = importdata(path_info);    % info_raw 1*1 struct �� .data .textdata(cell��ʽ�����ʽṹ�����ݺ��ı���
    id_raw = info_raw.textdata(2:end, 2);
    
    % Reconstruction
    info = struct();
    info.num = length(id_raw);
    info.filename = info_raw.textdata(2:end, 1);    %epoch000... =>filename
    info.session = info_raw.data(:, 1);
    info.condition = info_raw.data(:, 2);
    info.usage = info_raw.data(:, 3);
    info.id = zeros(info.num, 1);  % info.num �� 1�е�0   sub001 ->1
    for i = 1:info.num
        if strcmp(id_raw{i}, 'None')  %�Ƚ�s1 s2 ͬ1 ��ͬ0
            info.id(i) = NaN;
        else
            info.id(i) = str2double(regexp(id_raw{i}, '(?<=\w+)\d+', 'match')); % (?<=\w+)\d+   ɶ��˼������
        end
    end
end

% ʹ�� MATLAB ��������ʽ���� regexp ��һ���ַ��� id_raw{i} ����ȡ�������ֽ�β�����ַ�����
% ������ת��Ϊ���������͡�������˵����ִ�����²�����
% 
% ʹ��������ʽ (?<=\w+)\d+ ƥ���ַ����������ֽ�β�����ַ�����
% str2double ������ƥ�䵽�����ַ���ת��Ϊ���������͡�
% ���ش�������Ϊ���д���������
% id_raw{i} ��һ���ַ��������Ԫ�������еĵ� i ��Ԫ�ء�
% ͨ����ѭ����ʹ�øô��룬���Դ�������Ԫ�ز������ɸ�������ɵ�������
