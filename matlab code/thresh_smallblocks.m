function [y] = thresh_smallblocks(x)
if std2(x.data) < 1
    y = ones(size(x.data,1),size(x.data,2));
else
    y = im2bw(x.data,graythresh(x.data));
end