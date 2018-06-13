# Milestone 3 Submission

## Performance numbers
In this section, please attach the resource utilization and the cycle count of your application when running on the board: 
```bash 
# Please attach your report here
```
The full vcs logs for the SVM and NN are found in the files svm_vcssim.log and nn_vcssim.log, respectively.  In order to finish simulations in a reasonable time frame, we used a smaller number of images.  In the SVM, we used 600 images for training and 100 images for testing.  The simulation ran for 28113244 cycles, so the full set of images shouold be roughly 100 times longer.  The SVM has not been tested with paralellization yet.  The NN was tested with 200 testing images, and it ran for 71099856 cycles, and with the full set, we should expect 50 times as many cycles.  This design was parallelized.

## Design Choices
What design choices have you made to implement the hardware implementation? 
In order to take advantage of memory locality, we have designed all matrices so that when fetching from DRAM, we extract rows rather than columns.  Next, we have used Reduce controllers where appropriate to better optimize vector and matrix multiplications.  Additionally, we have parallelized loops where we were looping to compute probability values for each digit.  
