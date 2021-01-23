function generate_LR_Vimeo90K()
%% matlab code to genetate bicubic-downsampled for Vimeo90K dataset

up_scale = 16;
mod_scale = 16;
idx = 0;
filepaths = dir('/local/scratch/pmh64/datasets/vimeo_septuplet/sequences/*/*/*.png');
for i = 1 : length(filepaths)
    [~,imname,ext] = fileparts(filepaths(i).name);
    folder_path = filepaths(i).folder;
    save_LR_folder = strrep(folder_path,'vimeo_septuplet','vimeo_septuplet_matlabLRx16');
    if ~exist(save_LR_folder, 'dir')
        mkdir(save_LR_folder);
    end
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_result = sprintf('%d\t%s.\n', idx, imname);
        fprintf(str_result);
        % read image
        img = imread(fullfile(folder_path, [imname, ext]));
        img = im2double(img);
        % modcrop
        im_LR = modcrop(img, mod_scale);
        for l = 1 : log2(up_scale)
            im_LR = impyramid(im_LR, 'reduce');
        end
        if exist('save_LR_folder', 'var')
            imwrite(im_LR, fullfile(save_LR_folder, [imname, '.png']));
        end
    end
end
end

%% modcrop
function img = modcrop(img, modulo)
if size(img,3) == 1
    sz = size(img);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2));
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2),:);
end
end
