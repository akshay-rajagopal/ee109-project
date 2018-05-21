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
