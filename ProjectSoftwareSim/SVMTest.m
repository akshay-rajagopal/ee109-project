clearvars;clc;close all
load('mnist_training_data.mat')
load('mnist_test_data.mat')
n = 400;
rho = 1/25;
iters_train = 60000;
W = zeros(n,10);
% Training
for k = 1:iters_train
    x = images_train(:,k);
    for i = 1:10
        w = W(:,i);
        g_k = rho * w;
        y = -1;
        if (labels_train(k) == (i-1))
            y = 1;
        end
        if (1 - y*w.'*x > 0)
            g_k = g_k - x*y;
        end
        alpha_k = 0.0001;
        if(y == 1)
           alpha_k = alpha_k * 5; 
        end
        W(:,i) = w - alpha_k * g_k;
    end
end

incorrects = 0;
iters_test = 10000;
for j = 1:iters_test
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
end
error = incorrects/iters_test;
disp(['Accuracy: ' num2str(1-error)]);