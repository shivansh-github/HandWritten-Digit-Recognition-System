%% Machine Learning Author @Shivansh Bhasin


% FUNCTIONS INCLUDED
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m

clear ; close all; clc

%% Setup the parameters you will use for this part of the exercise
input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10
                          % (note that we have mapped "0" to label 10)

fprintf('HANDWRITTEN DIGIT RECOGNITION SYSTEM\n')
fprintf('author@Shivansh_Bhasin\n\n\n')
%% =========== Part 1: Loading and Visualizing Data =============

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('ex3data1.mat'); % training data stored in arrays X, y
m = size(X, 1);

% Randomly select 100 data points to display
rand_indices = randperm(m);
sel = X(rand_indices(1:100), :);

displayData(sel);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ============ Part 2a: Vectorize Logistic Regression ============
% Test case for lrCostFunction
fprintf('\nTesting lrCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\n Calculated Cost: %f\n', J);
% fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
% fprintf('Expected gradients:\n');
% fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('Program paused. Press enter to continue.\n');
pause;
%% ============ Part 2b: One-vs-All Training ============
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X);

fprintf('\nTraining Set Accuracy for one-vs-all classifier: %f\n', mean(double(pred == y)) * 100);

%------------------------------
fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
load('ex3weights.mat');

%% ================= Part 3: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);
fprintf('Predicting Accuracy For the Training Set...\n\n')
fprintf('\nTraining Set Accuracy for Neural Networks is: %f\n', mean(double(pred == y)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.

%  Randomly permute examples
rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
  fprintf('Running HandWritten Digit Recognition System...\n') 
   fprintf('\nNeural Network Prediction:The Predicted Digit is  %d (digit %d)\n', pred, mod(pred, 10));
    
    % Pause with quit option
    s = input('Paused - Press Enter to continue and E to exit:','s');
    if s == 'E'
      
      fprintf('\nThank You for using HandWritten Digit Recognition System\n\n\n\n\n\n')
      fprintf('Copyright @ShivanshBhasin@ShivanshBhasin@ShivanshBhasin@ShivanshBhasin@ShivanshBhasin@ShivanshBhasin@ShivanshBhasin')
      fprintf('\n\n\n\nFor Any Suggestions Email me :- shivanshbhasin0@gmail.com')
      fprintf('\n\nNote:-This is Summer Project of Shivansh Bhasin,Enrollment No:-75515002718,College:-M.S.I.T')
      break
    end
end

