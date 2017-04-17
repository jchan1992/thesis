% thresholding comparisons

clear all;
close all;
clc;

v = VideoReader('5.mp4');
frame = readFrame(v);
bgframe = rgb2gray(frame);


v.CurrentTime = 20;
frame = readFrame(v);
curframe = rgb2gray(frame);
figure;
imshow(curframe);

frame_gray = double(curframe) - double(bgframe);
[rows,columns] = size(frame_gray);

% thresh_otsu = graythresh(frame_gray)*255;
% frame_otsu = frame_gray;
% frame_otsu(frame_otsu>thresh_otsu) = 255;
% frame_otsu(frame_otsu<=thresh_otsu) = 0;
% figure;
% imshow(frame_otsu);
% 
% figure;
% [counts,bin] = imhist(frame_gray);
% thresh_triangles = thresh_triangle(counts,256);
% frame_triangle = frame_gray;
% frame_triangle(frame_triangle>thresh_triangles) = 255;
% frame_triangle(frame_triangle<=thresh_triangles) = 0;
% imshow(frame_triangle);
% 
% figure;
% frame_bradley = thresh_bradley(frame_gray);
% imshow(frame_bradley);
% 
% figure;
% frame_smallblocks = blockproc(frame_gray,[25 25],@thresh_smallblocks);
% imshow(frame_smallblocks);
% 
% figure;
% frame_sauvola = thresh_sauvola(frame_gray,[150 150]);
% imshow(frame_sauvola);
% 
% figure;
% frame_niblack = thresh_niblack(frame_gray,[25 25], -0.2,10);
% imshow(frame_niblack);
% 
% figure;
% frame_adaptive = thresh_adaptive(frame_gray,11,0.03,0);
% imshow(frame_adaptive);

figure;
frame_kde = frame_gray;
[lb,center] = thresh_kmeans(frame_gray);
[rows,cols] = size(frame_gray);
% frame_gray(lb==1) = 100;
% frame_gray(lb==2) = 150;
% maxnum = max(lb);
% maxnum = max(maxnum);
% frame_gray(lb==maxnum) = 255;
% frame_gray(lb~=maxnum) = 0;
% for i = 1:maxnum
%     lb(lb==i) = (255/i);
% end;
pixel_labels = reshape(lb,rows,cols);
imshow(pixel_labels,[]), title('image labeled by cluster index');
% imshow(frame_gray);
