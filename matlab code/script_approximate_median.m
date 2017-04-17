% author: Justin Chan z3373631
% final approximate median implementation

% set up matlab
clear all;
% close all;
% clc;

% set up constants
thresh = 30;
m = 2;
n = 2;

% set up figure
% figure('units','normalized','outerposition',[0 0 1 1])
figure(2);
clf;
hold on;
colormap(gray(256));
a1 = subplot(m,n,1);
a2 = subplot(m,n,2);
a3 = subplot(m,n,3);
a4 = subplot(m,n,4);
% tic;

% read in image
% disp('reading video...');
v = VideoReader('5.mp4');
% set video start time
% v.CurrentTime = 10;
% v.FrameRate

% get background frame
% disp('acquiring bg...');
bgFrame = readFrame(v);
bgFrame = rgb2gray(bgFrame);
[rows,columns] = size(bgFrame);

% % set up video recording
rec = VideoWriter('rec1.avi');
open(rec);

% setup plots
image(bgFrame,'Parent',a1);
% loop through video
while hasFrame(v) && v.CurrentTime<10
    frame = getframe(gcf);
    writeVideo(rec,frame);
%     disp('analysing video...');
    vidFrame = readFrame(v);
    vidFrame = rgb2gray(vidFrame);
    imagesc(vidFrame,'Parent',a2);
    title('video stream');
%      bsFrame = imabsdiff(vidFrame,bgFrame);
    bsFrame = abs(double(vidFrame) - double(bgFrame));
%     for i = 1: rows
%         for j = 1: columns
%             if bsFrame(i,j) > thresh ;
%                 mask(i,j) = 255;
%             else
%                 mask(i,j) = 0;
%             end
% %             background model updating
%             if vidFrame(i,j)> bgFrame(i,j)
%                 bgFrame(i,j) = bgFrame(i,j) + 1;
%             elseif vidFrame(i,j) < bgFrame(i,j)
%                 bgFrame(i,j) = bgFrame(i,j) - 1;
%             end
%         end
%     end
    mask = bsFrame;
    mask(mask>thresh) = 255;
    mask(mask<=thresh) = 0;
         for i = 1:rows*columns               
                if vidFrame(i)> bgFrame(i)
                    bgFrame(i) = bgFrame(i) + 1;
                elseif vidFrame(i) < bgFrame(i)
                    bgFrame(i) = bgFrame(i) - 1;
                end
        end
    
    imagesc(bgFrame,'Parent',a1);
    imagesc(bsFrame,'Parent',a3);    
    imagesc(mask,'Parent',a4);
    title(a1,'background');
    title(a2,'video stream');
    title(a3,'approximate median');
    title(a4,'threshold');
    drawnow;
    
%     pause(1/v.FrameRate);
end

close(rec);
disp('completed');
toc