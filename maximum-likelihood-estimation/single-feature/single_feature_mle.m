clear;
close all;
clc;

% Load the training set into the workspace from the given .mat file
load('../../data/train/TrainingSamplesDCT_8.mat');

% Ignore the first element in each training example because it is
% coefficient of the max value. Therefore find the max value after ignoring
% the first coefficient to obtain the 2nd largest value.(Features)
% The find command is used to obtain the index of the 2nd max value which is
% then stored in the feature matrix for FG and BG
[i, features_BG] = find(TrainsampleDCT_BG == max(TrainsampleDCT_BG(:,2:64),[],2));
[i, features_FG] = find(TrainsampleDCT_FG == max(TrainsampleDCT_FG(:,2:64),[],2));

[count_features_FG] = histcounts(features_FG);
count_features_FG = [zeros(1,1),count_features_FG,zeros(1,63-length(count_features_FG))];
prob_features_FG = count_features_FG/sum(count_features_FG);

[count_features_BG] = histcounts(features_BG);
count_features_BG = [zeros(1,1),count_features_BG,zeros(1,63-length(count_features_BG))];
prob_features_BG = count_features_BG/sum(count_features_BG);


% Finding the Prior probabilities
% The total number of features for FG and BG are found.
total_features_BG = length(features_BG);
total_features_FG = length(features_FG);
total_features = [total_features_BG, total_features_FG];
class = [1,2];
figure;
subplot(2,1,1);
bar(class,total_features);
title('Histogram plot for FG and BG');
ylabel('Number of features');
xlabel('Classs BG and FG');
% The respective probabilities are computed
prob_FG = total_features_FG/sum(total_features);
prob_BG = total_features_BG/sum(total_features);
prob_class = [prob_BG,prob_FG];
subplot(2,1,2);
bar(class,prob_class);
title('Prior Probabilities');
ylabel('Prior Probabilities');
xlabel('Classs BG and FG');
disp(' The prior PDF graph');
disp(' The prior probability_FG');
disp(prob_FG);
disp(' The prior probability_BG');
disp(prob_BG);
disp(' Press Enter to contiue');
pause;


% Finding the class conditional probabilities
subplot(2,1,1);
% The following is computed and can be plotted with the help of the 'bar'
% function, instead here we used the inbuilt normalization function in the
% histogram.
bin_edges = 0.5:1:64.5;
histogram(features_BG,bin_edges,'Normalization','pdf');
title('Class:Background (Grass)');
xlabel('Features X for BG (grass)');
ylabel('ccd PX|Y (x|grass)');
subplot(2,1,2);
% The following is computed and can be plotted with the help of the 'bar'
% function, instead here we used the inbuilt normalization function in the
% histogram.

histogram(features_FG,bin_edges,'Normalization','probability');
title('Class : Foreground (Cheetah)');
xlabel('Features X for FG (cheetah)');
ylabel('ccd PX|Y (x|cheetah) ');
disp('The class conditional probabilities graph');
disp('Press enter to continue');
pause;
% Read the groundtruth image mask
required_mask = imread('cheetah_mask.bmp');
required_mask = im2double(required_mask);

% Read the input image that is to be classified
[A,map] = imread('cheetah.bmp');
% Convert it to a range between [0 1] because the training data in in that
% range
A = im2double(A);
% Obtain the size of the image A
[img_row, img_col] = size(A);
% This matrix is used to store the 8x8 blocks of the image in each
% iteration
mat_64_D=zeros(8,8);
%This vector is used to store the 64 elements of a single block after the
%elements are zigzag classified
B = zeros(64,1);
% This matrix has rows corresponding the the zigzag classified 64 elements
% for each block. THe columns correspond to the number of blocks
mat_append = zeros(img_col-7*img_row-7,64);

test_op_img = zeros(img_row,img_col);
count_gx0_given_y1 = 0;
count_gx1_given_y1 = 0;
count_gx1_given_y0 = 0;
count_gx0_given_y0 = 0;
% Sliding Window Code
% This code initializes the head of the sliding window vector to the left
% most pixel and it slides along the entire row till it reaches the 
%'img_col-7'th column and the shifts to the next row and the procedure
%repeats. Each 8x8 block of data read into 'mat_64_D' is then zigzg
%classified into a vector B, and this is appended during each iteration to
%the matrix mat_append
% 
for row= 1:img_row-7
    for col = 1:img_col-7
        %Obtain the 8x8 block of the image
        mat_64_D = A(row:row+7,col:col+7);
        %Compute DCT
        mat_64_D = dct2(mat_64_D);
        % Find its absolute value
        mat_64_D = abs(mat_64_D);
        
        % ZigZag Code Start
        %This vector is used to store the 64 elements of a single block after the
        %elements are zigzag classified
        B = zeros(64,1);
        index = 1;
        for repeat = 1:8
        i = repeat;
        j = 1;
        
        while(i>0 && j > 0)
            if (mod(repeat,2)==1)
                B(index,1)=mat_64_D(i,j);
            elseif (mod(repeat,2)==0)
                B(index,1)=mat_64_D(j,i);    
            end
            i = i - 1;
            j = j + 1;
            index = index + 1;
        end
        end
        for repeat = 2:8
            i = repeat;
            j = 8;
            while(i<=8 && j <=8)
                if (mod(repeat,2)==1)
                    B(index,1)=mat_64_D(i,j);
                elseif (mod(repeat,2)==0)
                    B(index,1)= mat_64_D(j,i);
                end
                i = i + 1;
                j = j - 1;
                index = index + 1;
            end    
        end
        % Zigzag Code End
        
        B = B';
       
    % Obtaining the coefficient of the 2nd largest DCT values in each row 
    % This is a feature. Cummulative of everything forms the Feature Matrix     
        test_op_img(row,col) = find(B==max(B(:,2:64)));
        if(prob_features_BG(test_op_img(row,col))*(prob_BG) >= prob_features_FG(test_op_img(row,col))*(prob_FG))
           test_op_img(row,col)=0;
           
           
        else
            test_op_img(row,col)=1;
           
        end
        
        
        %mat_append = [ mat_append ; B' ];
    end
end
figure;
% Display The Final Classified Cheetah Mask
imshow(test_op_img);


for row = 1:img_row
    for col = 1:img_col
            if(required_mask(row,col) == 1)
                if(test_op_img(row,col) == 1)
                    count_gx1_given_y1 = count_gx1_given_y1 + 1;
            
                else
                   count_gx0_given_y1 = count_gx0_given_y1 + 1;
                end
            end
           

            if(required_mask(row,col) == 0)
                if(test_op_img(row,col) == 0)
                count_gx0_given_y0 = count_gx0_given_y0 + 1;
                else 
                count_gx1_given_y0 = count_gx1_given_y0 + 1;
                end
            end
    end
end


prob_gx0_given_y1 = count_gx0_given_y1 / (count_gx0_given_y1 + count_gx1_given_y1)
prob_gx1_given_y0 = count_gx1_given_y0 / (count_gx1_given_y0 + count_gx0_given_y0)


prob_error = prob_gx0_given_y1*prob_FG + prob_gx1_given_y0*prob_BG 
%[i, features_input] = find( mat_append == max(mat_append(:,2:64),[],2));














