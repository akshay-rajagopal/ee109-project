# Milestone 1 Submission

Akshay Rajagopal and Andrea Ramirez

## Application Description: 
```bash
# Please describe the application you want to work on here
```
We are training a series of Support Vector Machines in order to classify 
handwritten digits.  Our implementation consists of 10 SVMs, one per image, 
and we classify images by choosing the SVM that returns the highest score 
for that image.  We are using the MNIST dataset, which consists of 20-by-20
images of digits.  We are using online learning via a stochastic
subgradient method.  Our goal is to create a hardware implementation fast enough
to support large streams of incoming data.

## Software Implementation: 
```bash
# Please add your implementation of the application in software here. If you use github / bitbucket for version control, you can give a link to your repo. 
```
clearvars;clc;close all
load('mnist_training_data.mat')
load('mnist_test_data.mat')
n = 400;
%randn('state',0);
%w_true = randn(n,1); % 'true' weight vector
rho = 1/4;
iters_train = 60000;
W = zeros(n,10);
%w = zeros(n,1);
%error_probs = zeros(1,iters);
%diffs = zeros(1,iters);
for k = 1:iters_train
    x = images_train(:,k);
    for i = 1:10
        w = W(:,i);
        g_k = rho * w;
        %x = images_train(:,k);
        y = -1;
        if (labels_train(k) == (i-1))
            y = 1;
        end
        if (1 - y*w.'*x > 0)
            g_k = g_k - x*y;
        end
        alpha_k = 2/(k);
        if(y == 1)
           alpha_k = alpha_k * 5; 
        end
        W(:,i) = w - alpha_k * g_k;
        %diffs(k) = 1/iters*sum(pos(ones(1,iters)-w.'*xys)) + rho/2*(w.'*w) - f_opt;
        %error_probs(k) = 1 - 1/iters*sum(pos(sign(w.'*xys)));
    end
end
%error_guesses = zeros(10,1);
incorrects = 0;
for j = 1:10000
   x = images_test(:,j);
   maxval = W(:,1).'*x; maxind = 0;
   for i = 2:10
       if (W(:,i).'*x > maxval)
          maxval = W(:,i).'*x ;
          maxind = i-1;
       end
   end
   if (maxind ~= labels_test(j))
      incorrects = incorrects + 1; 
   end
%    for i = 1:10
%        y = -1;
%        if (labels_test(j) == (i-1))
%            y = 1;
%        end
%        if (y*W(:,i).'*x <= 0)
%            error_guesses(i) = error_guesses(i) + 1;
%        end
%    end
end
%error = error_guesses/10000;
error = incorrects/10000;
disp(['Accuracy: ' num2str(1-error)]);
%semilogy(1:iters,diffs); title('Optimality Gap');
%xlabel('k'); ylabel('log(f(x^{(k)}) - f*)');
%figure
%plot(1:iters,error_probs); title('Classifier Error Probability');
%xlabel('k');

## Setup Instructions:
```bash
# Please add instructions for running your software demo here
```
The MNIST training and testing images can be found as .mat files in the 
ProjectSoftwareSim folder on tucson.  These should be placed in MATLAB's 
working directory.  Then, simply run the script from above.  MATLAB
will print the percentage of correct guesses on the console.
