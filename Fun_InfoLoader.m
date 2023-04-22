function [info] = Fun_InfoLoader(path_info)
    % Load information of dataset from the given path.
    % param:
    %       path_info: the path of information file (.csv)
    % return:
    %       info: struct information (filename, id, session, condition, usage, num)
    
    % Import information
    info_raw = importdata(path_info);    % info_raw 1*1 struct 用 .data .textdata(cell格式）访问结构的数据和文本域
    id_raw = info_raw.textdata(2:end, 2);
    
    % Reconstruction
    info = struct();
    info.num = length(id_raw);
    info.filename = info_raw.textdata(2:end, 1);    %epoch000... =>filename
    info.session = info_raw.data(:, 1);
    info.condition = info_raw.data(:, 2);
    info.usage = info_raw.data(:, 3);
    info.id = zeros(info.num, 1);  % info.num 列 1行的0   sub001 ->1
    for i = 1:info.num
        if strcmp(id_raw{i}, 'None')  %比较s1 s2 同1 不同0
            info.id(i) = NaN;
        else
            info.id(i) = str2double(regexp(id_raw{i}, '(?<=\w+)\d+', 'match')); % (?<=\w+)\d+   啥意思？？？
        end
    end
end

% 使用 MATLAB 的正则表达式函数 regexp 从一个字符串 id_raw{i} 中提取出以数字结尾的子字符串，
% 并将其转换为浮点数类型。具体来说，它执行以下操作：
% 
% 使用正则表达式 (?<=\w+)\d+ 匹配字符串中以数字结尾的子字符串。
% str2double 函数将匹配到的子字符串转换为浮点数类型。
% 返回处理结果作为该行代码的输出。
% id_raw{i} 是一个字符串数组或单元格数组中的第 i 个元素。
% 通过在循环中使用该代码，可以处理所有元素并返回由浮点数组成的向量。
