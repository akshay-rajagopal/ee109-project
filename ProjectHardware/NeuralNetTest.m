images_labels = csvread('mnist_test_labels_28.csv');
images_test = csvread('mnist_test_images_28.csv');
W1 = csvread('W1.csv');
W2 = csvread('W2.csv');
b1 = csvread('b1.csv');
b2 = csvread('b2.csv');
errors = 0;
for i = 1
   img = images_test(i,:); 
   int = pos(img * W1.' + b1); %pos is the relu
   pred = int*W2.' + b2;
   [m, ind] = max(pred);
   if (ind-1 ~= images_labels(i))
      errors = errors + 1; 
   end
end
disp(1 - errors/10000);