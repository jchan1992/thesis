% author: Justin Chan z3373631
% final mog
% needs to be improved

% set up matlab
clear all;
% close all;
% clc;

% set up constants
thresh = 30;
l = 2;
n = 2;

% set up figure
% figure('units','normalized','outerposition',[0 0 1 1])
figure(3);
clf;
hold on;
colormap(gray(256));
a1 = subplot(l,n,1);
a2 = subplot(l,n,2);
a3 = subplot(l,n,3);
a4 = subplot(l,n,4);
tic;

% read in image
% disp('reading video...');
v = VideoReader('5.mp4');
% set video start time
v.CurrentTime = 1;
% v.FrameRate

% get background frame
% disp('acquiring bg...');
bgFrame = readFrame(v);
bgFrame = rgb2gray(bgFrame);
[height,width] = size(bgFrame);

% addition for mog
fgModel = zeros(height, width);
bgModel = zeros(height, width);

% mog variables

C = 3;                                  % number of gaussian components (typically 3-5)
M = 3;                                  % number of background components
D = 2.5;                                % positive deviation threshold
alpha = 0.001;                           % learning rate (between 0 and 1) (from paper 0.01)
thresh = 0.25;                          % foreground threshold (0.25 or 0.75 in paper)
sd_init = 6;                            % initial standard deviation (for new components) var = 36 in paper
weight = zeros(height,width,C);         % initialize weights array
mean = zeros(height,width,C);           % pixel means
sd = zeros(height,width,C);             % pixel standard deviations
u_diff = zeros(height,width,C);         % difference of each pixel from mean
p = alpha/(1/C);                        % initial p variable (used to update mean and sd)
rank = zeros(1,C);                      % rank of components (w/sd)

% --------------------- initialize component means and weights -----------

pixel_depth = 8;                        % 8-bit resolution
pixel_range = 2^pixel_depth -1;         % pixel range (# of possible values)

for i=1:height
    for j=1:width
        for k=1:C
            mean(i,j,k) = rand*pixel_range;     % means random (0-255)
            weight(i,j,k) = 1/C;                % weights uniformly dist
            sd(i,j,k) = sd_init;                % initialize to sd_init
            
        end
    end
end


% set up video recording
rec = VideoWriter('rec2.avi');
open(rec);

% setup plots
image(bgFrame,'Parent',a1);

% loop through video
while hasFrame(v) && v.CurrentTime < 10
    frame = getframe(gcf);
    writeVideo(rec,frame);
%     disp('analysing video...');

    vidFrame = readFrame(v);
    vidFrame = rgb2gray(vidFrame);
    imagesc(vidFrame,'Parent',a2);
    title('video stream');
    
 % calculate difference of pixel values from mean
    for l=1:C
        u_diff(:,:,l) = abs(double(vidFrame) - double(mean(:,:,l)));
    end
     
    % update gaussian components for each pixel
    for i=1:height
        for j=1:width       
            match = 0;
            for k=1:C                       
                if (abs(u_diff(i,j,k)) <= D*sd(i,j,k))       % pixel matches component
                    match = 1;                          % variable to signal component match
                    % update weights, mean, sd, p
                    weight(i,j,k) = (1-alpha)*weight(i,j,k) + alpha;
                    p = alpha/weight(i,j,k);                  
                    mean(i,j,k) = (1-p)*mean(i,j,k) + p*double(vidFrame(i,j));
                    sd(i,j,k) = sqrt((1-p)*(sd(i,j,k)^2) + p*((double(vidFrame(i,j)) - mean(i,j,k)))^2);
                else                                    % pixel doesn't match component
                    weight(i,j,k) = (1-alpha)*weight(i,j,k);      % weight slighly decreases
                    
                end
            end
            
            weight(i,j,:) = weight(i,j,:)./sum(weight(i,j,:));
            bgModel(i,j)=0;
            for k=1:C
                bgModel(i,j) = bgModel(i,j)+ mean(i,j,k)*weight(i,j,k);
            end
            
            % if no components match, create new component
            if (match == 0)
                [min_w, min_w_index] = min(weight(i,j,:));  
                mean(i,j,min_w_index) = double(vidFrame(i,j));
                sd(i,j,min_w_index) = sd_init;
            end

            rank = weight(i,j,:)./sd(i,j,:);             % calculate component rank
            rank_ind = [1:1:C];
            
            % sort rank values
            for k=2:C               
                for l=1:(k-1)
                    
                    if (rank(:,:,k) > rank(:,:,l))                     
                        % swap max values
                        rank_temp = rank(:,:,l);  
                        rank(:,:,l) = rank(:,:,k);
                        rank(:,:,k) = rank_temp;
                        
                        % swap max index values
                        rank_ind_temp = rank_ind(l);  
                        rank_ind(l) = rank_ind(k);
                        rank_ind(k) = rank_ind_temp;    

                    end
                end
            end
            
            % calculate foreground
            match = 0;
            k=1;
            
            fgModel(i,j) = 0;
            while ((match == 0)&&(k<=M))

                if (weight(i,j,rank_ind(k)) >= thresh)
                    if (abs(u_diff(i,j,rank_ind(k))) <= D*sd(i,j,rank_ind(k)))
                        fgModel(i,j) = 0;
                        mask(i,j) = 0;
                        match = 1;
                    else
                        fgModel(i,j) = vidFrame(i,j);
                        mask(i,j) = 255;
                    end
                end
                k = k+1;
            end
        end
    end
    
    imagesc(bgModel,'Parent',a1);
    imagesc(fgModel,'Parent',a3);    
    imagesc(mask,'Parent',a4);
    title(a1,'background');
    title(a2,'video stream');
    title(a3,'mog');
    title(a4,'threshold');
    drawnow;
    
%     pause(3/v.FrameRate);
% pause(1)
end

close(rec);
disp('completed');
toc