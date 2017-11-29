clear;
close all;
clc;

%% Section 1: Prior Probability
% Load the training to the workspace
load('../../data/train/TrainingSamplesDCT_8_new.mat')

% Calculating the number of training exmaples for the BG and FG classes
[ no_training_samples_BG, no_features_BG] = size(TrainsampleDCT_BG);
[ no_training_samples_FG, no_features_FG] = size(TrainsampleDCT_FG);
no_total_training_samples =  [no_training_samples_BG , no_training_samples_FG ] ;

% Computing the prior probability of the classes using the fact for a
% multinomial distribution is number of training examples in a class 
% divided the total number of training examples.

P_BG = no_training_samples_BG / sum(no_total_training_samples);
P_FG = no_training_samples_FG / sum(no_total_training_samples);
prob_class = [P_BG, P_FG];
% 1 represents BG(Grass) and 2 represent FG(Cheetah)0
class = [1 2];

figure;
subplot(2,1,1);
bar(class,no_total_training_samples);
title('Histogram plot for FG and BG');
ylabel('Number of features');
xlabel('Classs BG and FG');

subplot(2,1,2);
bar(class,prob_class);
title('Prior Probabilities');
ylabel('Prior Probabilities');
xlabel('Classs BG and FG');

disp(' The prior probability of BG(Grass) class P(Grass) :');
disp(P_BG);
disp(' The prior probability of FG(Cheetah) class P(Cheetah) :');
disp(P_FG);

%% Section 2 : Marginal Densities for the two classes.
% Estimates of mean and covariance
 
% Background class ( GRASS) 
% Initialising the covariance matrix (64 * 64 ) to zero
% or the  (no_of_features) * (no_of_features) to zero.

 covariance_estimate_64_BG = zeros(no_features_BG,no_features_BG);

 % The estimate for each feature can be found as the mean of the individual
 % samples over the entire set of training examples
 % Converting it into a ( no_features * 1 ) matrix

 mean_estimate_64_BG = sum(TrainsampleDCT_BG) / no_training_samples_BG;
 mean_estimate_64_BG = mean_estimate_64_BG';
 
 
 
 % Finding the covariance matrix for each training example and then summing
 % them all up. After which the average covariance matrix is found out.
 for training_example = 1 : no_training_samples_BG
    
     X =(TrainsampleDCT_BG(training_example,:))';
     covariance_estimate_64_BG = covariance_estimate_64_BG ...
         + ((X - mean_estimate_64_BG) * (X - mean_estimate_64_BG)');
     
 end
 
 covariance_estimate_64_BG =  covariance_estimate_64_BG / no_training_samples_BG;
 alpha_64_BG = alpha(covariance_estimate_64_BG,P_BG,no_features_BG);
 inv_covariance_estimate_64_BG = inv(covariance_estimate_64_BG);

 
 % Foreground class (Cheetah)
 
 % Initialising the covariance matrix (64 * 64 ) to zero
 % or the  (no_of_features) * (no_of_features) to zero.
 covariance_estimate_64_FG = zeros(no_features_FG,no_features_FG);
 
 % The estimate for each feature can be found as the mean of the individual
 % samples over the entire set of training examples
 % Converting it into a ( no_features * 1 ) matrix
 mean_estimate_64_FG = sum(TrainsampleDCT_FG) / no_training_samples_FG ;
 mean_estimate_64_FG = mean_estimate_64_FG';
 
 % Finding the covariance matrix for each training example and then summing
 % them all up. After which the average covariance matrix is found out.
 
 for training_example = 1 : no_training_samples_FG
     
     X = (TrainsampleDCT_FG(training_example,:))';
     covariance_estimate_64_FG = covariance_estimate_64_FG ...
         + ((X - mean_estimate_64_FG)*(X-mean_estimate_64_FG)');
 end
 
 covariance_estimate_64_FG = covariance_estimate_64_FG / no_training_samples_FG;
 alpha_64_FG = alpha(covariance_estimate_64_FG,P_FG,no_features_FG);
 inv_covariance_estimate_64_FG = inv(covariance_estimate_64_FG);
     
  %% Section Best 8 plots and Worst 8 plots
  best_eight_features =  [1,12,19,20,24,34,35,38]; 
  worst_eight_features = [1:1:12];%[63,37,55,51,48,56,64,35];
  
  figure;
  title('Best Eight Features Marginal PDFs');
  for count = 1:8
  
      i = best_eight_features(count);
      
      y_BG = exp(((x-mean_estimate_64_BG(i)).^2)/(-2*covariance_estimate_64_BG(i,i))) ...
     *(1/(sqrt(2*pi*covariance_estimate_64_BG(i,i))));
        
        subplot(2,4,count);
        plot (x,y_BG,'-m', ...
               'LineWidth',2);
        hold on;
 
 
        y_FG = exp(((x-mean_estimate_64_FG(i)).^2)/(-2*covariance_estimate_64_FG(i,i))) ...
     *(1/(sqrt(2*pi*covariance_estimate_64_FG(i,i))));
     
     subplot(2,4,count);
     plot (x,y_FG,'-r', ...
               'LineWidth',2);
     
    
     title(sprintf('Marginal PDFs  feature %d ',i));
     ylabel ('P_{XK/Y}(xk/y)');
     
        min_FG = (mean_estimate_64_FG(i) - ( sqrt(covariance_estimate_64_FG(i,i)) * 3));
     max_FG = (mean_estimate_64_FG(i) + ( sqrt(covariance_estimate_64_FG(i,i)) * 3));
     
     min_BG = (mean_estimate_64_BG(i) - ( sqrt(covariance_estimate_64_BG(i,i)) *3));
     max_BG = (mean_estimate_64_BG(i) + ( sqrt(covariance_estimate_64_BG(i,i)) *3));
     
     %pause;
     axis([ min(min_FG,min_BG) ,...
            max(max_FG,max_BG), ...
            min([y_FG , y_BG]), ...
            max([y_FG , y_BG]) ]);
     hold off;
     
  end
  
  diff_array_prop = zeros(64,1);
  count_global  =1
  
  
  figure;
  title('Worst Eight Features Marginal PDFs');
  for count = 1:12
  
      i = worst_eight_features(count);
      
      y_BG = exp(((x-mean_estimate_64_BG(i)).^2)/(-2*covariance_estimate_64_BG(i,i))) ...
     *(1/(sqrt(2*pi*covariance_estimate_64_BG(i,i))));
        
        subplot(3,4,count);
        plot (x,y_BG,'-m', ...
               'LineWidth',2);
        hold on;
 
 
        y_FG = exp(((x-mean_estimate_64_FG(i)).^2)/(-2*covariance_estimate_64_FG(i,i))) ...
     *(1/(sqrt(2*pi*covariance_estimate_64_FG(i,i))));
     
     subplot(3,4,count);
     plot (x,y_FG,'-r', ...
               'LineWidth',2);
     
    
     title(sprintf('Marginal PDFs  feature %d ',i));
     ylabel ('P_{XK/Y}(xk/y)');
     
        min_FG = (mean_estimate_64_FG(i) - ( sqrt(covariance_estimate_64_FG(i,i)) * 3));
     max_FG = (mean_estimate_64_FG(i) + ( sqrt(covariance_estimate_64_FG(i,i)) * 3));
     
     min_BG = (mean_estimate_64_BG(i) - ( sqrt(covariance_estimate_64_BG(i,i)) *3));
     max_BG = (mean_estimate_64_BG(i) + ( sqrt(covariance_estimate_64_BG(i,i)) *3));
     
     %pause;
     axis([ min(min_FG,min_BG) ,...
            max(max_FG,max_BG), ...
            min([y_FG , y_BG]), ...
            max([y_FG , y_BG]) ]);
     hold off;
     diff_array_prop(count_global,1) = y_FG/(y_FG + y_BG);
     count_global =count_global + 1;
  end
  
  
  worst_eight_features = [13:1:24];
   figure;
  title('Worst Eight Features Marginal PDFs');
  for count = 1:12
  
      i = worst_eight_features(count);
      
      y_BG = exp(((x-mean_estimate_64_BG(i)).^2)/(-2*covariance_estimate_64_BG(i,i))) ...
     *(1/(sqrt(2*pi*covariance_estimate_64_BG(i,i))));
        
        subplot(3,4,count);
        plot (x,y_BG,'-m', ...
               'LineWidth',2);
        hold on;
 
 
        y_FG = exp(((x-mean_estimate_64_FG(i)).^2)/(-2*covariance_estimate_64_FG(i,i))) ...
     *(1/(sqrt(2*pi*covariance_estimate_64_FG(i,i))));
     
     subplot(3,4,count);
     plot (x,y_FG,'-r', ...
               'LineWidth',2);
     
    
     title(sprintf('Marginal PDFs  feature %d ',i));
     ylabel ('P_{XK/Y}(xk/y)');
     
        min_FG = (mean_estimate_64_FG(i) - ( sqrt(covariance_estimate_64_FG(i,i)) * 3));
     max_FG = (mean_estimate_64_FG(i) + ( sqrt(covariance_estimate_64_FG(i,i)) * 3));
     
     min_BG = (mean_estimate_64_BG(i) - ( sqrt(covariance_estimate_64_BG(i,i)) *3));
     max_BG = (mean_estimate_64_BG(i) + ( sqrt(covariance_estimate_64_BG(i,i)) *3));
     
     %pause;
     axis([ min(min_FG,min_BG) ,...
            max(max_FG,max_BG), ...
            min([y_FG , y_BG]), ...
            max([y_FG , y_BG]) ]);
     hold off;
     diff_array_prop(count_global,1) = y_FG/(y_FG + y_BG);
     count_global =count_global + 1;
  end
  
  
    worst_eight_features = [25:1:36];
   figure;
  title('Worst Eight Features Marginal PDFs');
  for count = 1:12
  
      i = worst_eight_features(count);
      
      y_BG = exp(((x-mean_estimate_64_BG(i)).^2)/(-2*covariance_estimate_64_BG(i,i))) ...
     *(1/(sqrt(2*pi*covariance_estimate_64_BG(i,i))));
        
        subplot(3,4,count);
        plot (x,y_BG,'-m', ...
               'LineWidth',2);
        hold on;
 
 
        y_FG = exp(((x-mean_estimate_64_FG(i)).^2)/(-2*covariance_estimate_64_FG(i,i))) ...
     *(1/(sqrt(2*pi*covariance_estimate_64_FG(i,i))));
     
     subplot(3,4,count);
     plot (x,y_FG,'-r', ...
               'LineWidth',2);
     
    
     title(sprintf('Marginal PDFs  feature %d ',i));
     ylabel ('P_{XK/Y}(xk/y)');
     
        min_FG = (mean_estimate_64_FG(i) - ( sqrt(covariance_estimate_64_FG(i,i)) * 3));
     max_FG = (mean_estimate_64_FG(i) + ( sqrt(covariance_estimate_64_FG(i,i)) * 3));
     
     min_BG = (mean_estimate_64_BG(i) - ( sqrt(covariance_estimate_64_BG(i,i)) *3));
     max_BG = (mean_estimate_64_BG(i) + ( sqrt(covariance_estimate_64_BG(i,i)) *3));
     
     %pause;
     axis([ min(min_FG,min_BG) ,...
            max(max_FG,max_BG), ...
            min([y_FG , y_BG]), ...
            max([y_FG , y_BG]) ]);
     hold off;
     diff_array_prop(count_global,1) = y_FG/(y_FG + y_BG);
     count_global =count_global + 1;
  end
  
  
    worst_eight_features = [37:1:48];
   figure;
  title('Worst Eight Features Marginal PDFs');
  for count = 1:12
  
      i = worst_eight_features(count);
      
      y_BG = exp(((x-mean_estimate_64_BG(i)).^2)/(-2*covariance_estimate_64_BG(i,i))) ...
     *(1/(sqrt(2*pi*covariance_estimate_64_BG(i,i))));
        
        subplot(3,4,count);
        plot (x,y_BG,'-m', ...
               'LineWidth',2);
        hold on;
 
 
        y_FG = exp(((x-mean_estimate_64_FG(i)).^2)/(-2*covariance_estimate_64_FG(i,i))) ...
     *(1/(sqrt(2*pi*covariance_estimate_64_FG(i,i))));
     
     subplot(3,4,count);
     plot (x,y_FG,'-r', ...
               'LineWidth',2);
     
    
     title(sprintf('Marginal PDFs  feature %d ',i));
     ylabel ('P_{XK/Y}(xk/y)');
     
        min_FG = (mean_estimate_64_FG(i) - ( sqrt(covariance_estimate_64_FG(i,i)) * 3));
     max_FG = (mean_estimate_64_FG(i) + ( sqrt(covariance_estimate_64_FG(i,i)) * 3));
     
     min_BG = (mean_estimate_64_BG(i) - ( sqrt(covariance_estimate_64_BG(i,i)) *3));
     max_BG = (mean_estimate_64_BG(i) + ( sqrt(covariance_estimate_64_BG(i,i)) *3));
     
     %pause;
     axis([ min(min_FG,min_BG) ,...
            max(max_FG,max_BG), ...
            min([y_FG , y_BG]), ...
            max([y_FG , y_BG]) ]);
     hold off;
     diff_array_prop(count_global,1) = y_FG/(y_FG + y_BG);
     count_global =count_global + 1;
  end
  
    worst_eight_features = [49:1:64];
   figure;
  title('Worst Eight Features Marginal PDFs');
  for count = 1:16
  
      i = worst_eight_features(count);
      
      y_BG = exp(((x-mean_estimate_64_BG(i)).^2)/(-2*covariance_estimate_64_BG(i,i))) ...
     *(1/(sqrt(2*pi*covariance_estimate_64_BG(i,i))));
        
        subplot(4,4,count);
        plot (x,y_BG,'-m', ...
               'LineWidth',2);
        hold on;
 
 
        y_FG = exp(((x-mean_estimate_64_FG(i)).^2)/(-2*covariance_estimate_64_FG(i,i))) ...
     *(1/(sqrt(2*pi*covariance_estimate_64_FG(i,i))));
     
     subplot(4,4,count);
     plot (x,y_FG,'-r', ...
               'LineWidth',2);
     
    
     title(sprintf('Marginal PDFs  feature %d ',i));
     ylabel ('P_{XK/Y}(xk/y)');
     
        min_FG = (mean_estimate_64_FG(i) - ( sqrt(covariance_estimate_64_FG(i,i)) * 3));
     max_FG = (mean_estimate_64_FG(i) + ( sqrt(covariance_estimate_64_FG(i,i)) * 3));
     
     min_BG = (mean_estimate_64_BG(i) - ( sqrt(covariance_estimate_64_BG(i,i)) *3));
     max_BG = (mean_estimate_64_BG(i) + ( sqrt(covariance_estimate_64_BG(i,i)) *3));
     
     %pause;
     axis([ min(min_FG,min_BG) ,...
            max(max_FG,max_BG), ...
            min([y_FG , y_BG]), ...
            max([y_FG , y_BG]) ]);
     hold off;
     diff_array_prop(count_global,1) = y_FG/(y_FG + y_BG);
     count_global =count_global + 1;
  end
  
  
  [ diff_array_prop, ind ] = sort(diff_array_prop);
  diff_array_prop_t = [diff_array_prop , ind ];
 %% Section 3
 
 desired_mask = imread('cheetah_mask.bmp');
 desired_mask = im2double(desired_mask);
 
 
 [input_image, map] = imread('cheetah.bmp');
 input_image = im2double(input_image);
 
 [img_no_rows , img_no_cols] = size(input_image);
 mat_64_D = zeros(8,8);
 vec_64_D = zeros(64,1);
 
 generated_mask_64 = zeros (img_no_rows,img_no_cols);
 generated_mask_8 = zeros (img_no_rows,img_no_cols);
 
  diff_mean = abs(mean_estimate_64_BG - mean_estimate_64_FG);
 
 [diff_mean , index ] = sort(diff_mean);
 diff_mean_t = [diff_mean,index];

 
% best_eight_features = [12,19,20,24,34,35,38,1];% worst 18.5 %[2,3,4,62,63,64,59,5];
 
 
 transforming_matrix =  zeros( length(best_eight_features) , no_features_BG);
 
 for row = 1: size(transforming_matrix,1)
 
     transforming_matrix(row,best_eight_features(1,row)) = 1;
         
 end
 
 mean_eight_features_BG = transforming_matrix * mean_estimate_64_BG;
 mean_eight_features_FG = transforming_matrix * mean_estimate_64_FG;
 
 covar_eight_features_BG = transforming_matrix * covariance_estimate_64_BG * ...
                            transforming_matrix';                                            
 covar_eight_features_FG = transforming_matrix * covariance_estimate_64_FG * ...
                            transforming_matrix';
 
 alpha_8_FG = alpha(covar_eight_features_FG,P_FG,8);                                           
 alpha_8_BG = alpha(covar_eight_features_BG,P_BG,8);                  
  inv_covar_eight_features_FG = inv(covar_eight_features_FG);
  inv_covar_eight_features_BG =  inv(covar_eight_features_BG);
                        
 %% Section 4 Classification 64 features and 8 features
                      
 for row = 1:img_no_rows -7
     for col = 1:img_no_cols - 7
    
         mat_64_D = input_image(row:row+7,col:col+7);
         mat_64_D = dct2(mat_64_D);
        %------------------------------------------------------------------ 
        % ZigZag Code Start
        %------------------------------------------------------------------
        %This vector is used to store the 64 elements of a single block 
        %after the elements are zigzag classified
        
        vec_64_D = zeros(64,1);
        vec_8_D = zeros(8,1);
        index = 1;
        for repeat = 1:8
        i = repeat;
        j = 1;
        
        while(i>0 && j > 0)
            if (mod(repeat,2)==1)
                vec_64_D(index,1)=mat_64_D(i,j);
            elseif (mod(repeat,2)==0)
                vec_64_D(index,1)=mat_64_D(j,i);    
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
                    vec_64_D(index,1)=mat_64_D(i,j);
                elseif (mod(repeat,2)==0)
                    vec_64_D(index,1)= mat_64_D(j,i);
                end
                i = i + 1;
                j = j - 1;
                index = index + 1;
            end    
        end
        %------------------------------------------------------------------
        % Zigzag Code End
        %------------------------------------------------------------------
       
        %------------------------------------------------------------------
        % Finding classification mask for 64 features
        %------------------------------------------------------------------
        distance_alpha_BG = (mahalanobis_distance(vec_64_D,mean_estimate_64_BG,inv_covariance_estimate_64_BG)) ...
            + alpha_64_BG;
       
        distance_alpha_FG = (mahalanobis_distance(vec_64_D,mean_estimate_64_FG,inv_covariance_estimate_64_FG)) ...
            + alpha_64_FG;
       
       

        if( distance_alpha_BG <= distance_alpha_FG )
           
            generated_mask_64(row,col) = 0;
            
        else
            
            generated_mask_64(row,col) = 1;
           
        end
        
        %------------------------------------------------------------------
        % Finding classification mask for 64 featrues end
        %------------------------------------------------------------------
        
        %-------------------------------------------------------------------
        % Finding for best 8 featrues
        %-------------------------------------------------------------------
       
        
        for row_no = 1: size(transforming_matrix,1)
           
            vec_8_D(row_no,1) = vec_64_D(best_eight_features(1,row_no),1);
            
        end
        
        distance_alpha_BG_8 = (mahalanobis_distance(vec_8_D,mean_eight_features_BG,inv_covar_eight_features_BG)) ...
            + alpha_8_BG;
       
        distance_alpha_FG_8 = (mahalanobis_distance(vec_8_D,mean_eight_features_FG,inv_covar_eight_features_FG)) ...
            + alpha_8_FG;
        
        
         if( distance_alpha_BG_8 <= distance_alpha_FG_8 )
           
            generated_mask_8(row,col) = 0;
            
         else
            
            generated_mask_8(row,col) = 1;
           
         end   
        
         
        %------------------------------------------------------------------
        % Finding classified mask for best 8 featrues - END
        %------------------------------------------------------------------
     end
 end
  
 figure;
 imshow(generated_mask_64);
 title(' Classified mask for 64 features');
 
 figure;
 imshow(generated_mask_8);
 title(' Classified mask for 8 features');
 
 %% Section Probability Of Error - 64 feature mask
 
count_gx0_given_y1 = 0;
count_gx1_given_y1 = 0;
count_gx1_given_y0 = 0;
count_gx0_given_y0 = 0;

for row = 1:img_no_rows
    for col = 1:img_no_cols
            if(desired_mask(row,col) == 1)
                if(generated_mask_64(row,col) == 1)
                    count_gx1_given_y1 = count_gx1_given_y1 + 1;
            
                else
                   count_gx0_given_y1 = count_gx0_given_y1 + 1;
                end
            end
           

            if(desired_mask(row,col) == 0)
                if(generated_mask_64(row,col) == 0)
                count_gx0_given_y0 = count_gx0_given_y0 + 1;
                else 
                count_gx1_given_y0 = count_gx1_given_y0 + 1;
                end
            end
    end
end
 
prob_gx0_given_y1_64 = count_gx0_given_y1 / (count_gx0_given_y1 + count_gx1_given_y1)
prob_gx1_given_y0_64 = count_gx1_given_y0 / (count_gx1_given_y0 + count_gx0_given_y0)


prob_error_64 = prob_gx0_given_y1_64*P_FG + prob_gx1_given_y0_64*P_BG ;
fprintf('Probability Of Error 64 feature mask %d \n',prob_error_64);

%% Section Probability of Error 8 feature mask
  
count_gx0_given_y1 = 0;
count_gx1_given_y1 = 0;
count_gx1_given_y0 = 0;
count_gx0_given_y0 = 0;

for row = 1:img_no_rows 
    for col = 1:img_no_cols 
            if(desired_mask(row,col) == 1)      % Given that pixel is FG or white
                if(generated_mask_8(row,col) == 1)
                    count_gx1_given_y1 = count_gx1_given_y1 + 1;
            
                else
                   count_gx0_given_y1 = count_gx0_given_y1 + 1;
                end
            end
           

            if(desired_mask(row,col) == 0)      % Given that pixel is BG or black
                if(generated_mask_8(row,col) == 0)
                count_gx0_given_y0 = count_gx0_given_y0 + 1;
                else 
                count_gx1_given_y0 = count_gx1_given_y0 + 1;
                end
            end
    end
end
 
prob_gx0_given_y1_8 = count_gx0_given_y1 / (count_gx0_given_y1 + count_gx1_given_y1)
prob_gx1_given_y0_8 = count_gx1_given_y0 / (count_gx1_given_y0 + count_gx0_given_y0)

prob_error_8 = prob_gx0_given_y1_8*P_FG + prob_gx1_given_y0_8*P_BG;
fprintf('Probability Of Error 8 feature mask %d \n',prob_error_8);  



 