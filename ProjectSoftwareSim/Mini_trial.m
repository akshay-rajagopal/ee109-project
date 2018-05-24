clearvars;clc;close all
n = 4;
iters_train = 2;
%rho = 1/2;
rho = 1/25;
digits = 3;
W = zeros(n,digits);
images_train = [0.5 0.7; 0 0.3; 1 0; 0.2 0.5];
labels_train = [1 0];
images_test = [0; 0; 0.9; 0];
labels_test = [1];

for k = 1:iters_train
    x = images_train(:,k);
    for i = 1:digits
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
        alpha_k = 0.0001;
        if(y == 1)
           alpha_k = alpha_k * 5; 
        end
        W(:,i) = w - alpha_k * g_k;
        %diffs(k) = 1/iters*sum(pos(ones(1,iters)-w.'*xys)) + rho/2*(w.'*w) - f_opt;
        %error_probs(k) = 1 - 1/iters*sum(pos(sign(w.'*xys)));
    end
    disp('W = '); disp(W.');
end
incorrects = 0;
for j = 1
   x = images_test(:,j);
   maxval = W(:,1).'*x; maxind = 0;
   for i = 2:digits
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
disp(incorrects)