function fea = Fun_FeaExtract(data, param)
    % Extract features from EEG data.
    % param:
    %       data: EEG data in [n_channel×n_samples]
    %       param: params of preprocess {'fs', 'fea_type', 'n_fea', ...}
    % return:
    %       fea: features in [1×(n_channel×n_fea)]
    
    % Frame settings
    fs = param.fs;
    n_win = floor(1*fs);  % Num of samples in each frame  向下取整
    n_overlap = floor(0.5*fs); % Num of overlaped samples in two frames
    n_channel = size(data, 1);
    
    switch param.fea_type
        case 'psd'
            for idx_chan = 1:n_channel
                fea(idx_chan, :) = extract_psd(squeeze(data(idx_chan, :)), ...
                n_win, n_overlap, fs, param.filter_band(1), param.filter_band(2), param.n_fea);
            end
            fea = reshape(fea, 1, length(fea(:)));
        case 'plv'
            fea_raw = extract_plv(data,n_channel);
            fea = [];
            for idx_i = 1:n_channel
                for idx_j = idx_i+1:n_channel
                    fea = [fea, fea_raw(idx_i, idx_j)];
                end
            end
        case 'pli'
            fea_raw = extract_pli(data,n_channel);
            fea = [];
            for idx_i = 1:n_channel
                for idx_j = idx_i+1:n_channel
                    fea = [fea, fea_raw(idx_i, idx_j)];
                end
            end
        case 'wpli'
            fea_raw = extract_wpli(data,n_channel);
            fea=[];
            for idx_i = 1:n_channel
                for idx_j = idx_i+1:n_channel
                    fea = [fea, fea_raw(idx_i, idx_j)];
                end
            end
        case 'corr'   % bad
            fea_raw = corrcoef(data');
            fea=[];
            for idx_i = 1:n_channel
                for idx_j = idx_i+1:n_channel
                    fea = [fea, fea_raw(idx_i, idx_j)];
                end
            end
            
        otherwise
            error('Error feature type!');
    end

end

function psd = extract_psd(X, n_win, n_overlap, fs, f_low, f_high, n_band)
    % Extract power spectral density (PSD).
    % param:
    %       X: data in [1×n_samples]
    %       n_win: num of samples in each frame
    %       n_overlap: num of overlaped samples in two frames
    %       fs: sampling rate (Hz)
    %       f_low: lowest frequency of interested filtband
    %       f_high: highest frequency of interested filtband
    %       n_band: num of power bands
    % return:
    %       psd: psd features in (1×n_band)
    
    win = hamming(n_win);
    [pxx, f] = pwelch(X, win, n_overlap, n_win, fs);
    % Slice banks
    idx_freq = linspace(find((f-f_low)==min(abs(f-f_low))), ...
        find((f-f_high)==min(abs(f-f_high))), n_band+1);  
    idx_freq = floor(idx_freq); 
    for i = 1:n_band
        % Band power
        psd(i) = sum(pxx(idx_freq(i):idx_freq(i+1)));  
    end
    
end

function pli = extract_pli(X, ch) 
    % Extract phase lage index (pli).
    % param:
    %       X: data in [n_channel×n_samples]
    %       ch: n_channel
    % return:
    %       pli: pli features in (n_channel×n_channel)
    analytic_data = angle(hilbert(X));
    data_complex = analytic_data';
    pli = ones(ch,ch);   % =1 说明就是完全同步关系
    for ch1 = 1:ch-1
        for ch2 = ch1+1:ch
            pdiff = data_complex(:,ch1)-data_complex(:,ch2); % phase difference
            pli(ch1,ch2) = abs(mean(sign(sin(pdiff)))); % only count the asymmetry
            %pli(ch1,ch2) = abs(mean(sign(pdiff))); % only count the asymmetry
            pli(ch2,ch1) = pli(ch1,ch2);
        end
    end
end

function plv = extract_plv(X, ch)
    % Extract phase lock value (plv).
    % param:
    %       X: data in [n_channel×n_samples]
    %       ch: n_channel
    % return:
    %       plv: plv features in (n_channel×n_channel)
    analytic_data = angle(hilbert(X));
    data_complex = analytic_data';
    plv = ones(ch,ch);
    for ch1 = 1:ch-1
        for ch2 = ch1+1:ch
            pdiff = data_complex(:,ch1)-data_complex(:,ch2);  % phase difference
            plv(ch1,ch2) = abs(mean(exp(1i*(pdiff)))); % only count the asymmetry
            plv(ch2,ch1) = plv(ch1,ch2);
        end
    end
end

function wpli = extract_wpli(X, ch) 
    % Extract weight phase lag index (wpli).
    % param:
    %       X: data in [n_channel×n_samples]
    %       ch: n_channel
    % return:
    %       wpli: wpli features in (n_channel×n_channel)
    datac = hilbert(X);
    wpli = ones(ch,ch);
    for ch1 = 1:ch-1
        for ch2 = ch1+1:ch
            data_img = angle(datac(:, ch1)./datac(:, ch2));  
			% phase difference  angle(a)-angle(b)=angle(a/b)=angle(a*conj(b))
            wpli(ch1,ch2) = abs(sum(abs(data_img).*sign(data_img))/sum(abs(data_img)));  
			%abs(mean(abs(imag(a*conj(b)))))/mean(abs(...)); 
            wpli(ch2,ch1) = wpli(ch1,ch2); 
        end
    end
    for i=1:ch
        wpli(i,i)=1;
    end
end






