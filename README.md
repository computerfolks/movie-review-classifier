# Movie Review Classifier

There are two programs in this project. The first program, pre_process.py, performs pre-processing on a set of movie reviews. It calculates word frequencies for positive and negative examples, and then scans each movie review and converts it into a vector of features with its label. The second program, LR.py, builds linear regression model parameters using the randomized training examples, cross-entropy loss, and stochastic gradient descent. It then tests the model parameters against training examples, generating a prediction for each example.

pre_process.py first generates frequencies. It passes a negative output file (word frequencies for the negative reviews), positive output file, differential output file (word frequency in positive reviews minus word frequency in negative reviews), and total output file  (combined positive/negative word frequency) into the generate_frequencies() function. Each file has one word on each line, sorted by most frequent words (or highest absolute value of differential in the case of the differential output file).

The program then finds the number of words the user requests, starting from the highest values in the sorted file. The program clears files which use the ‘a’ rather than ‘w’ file writing feature, and then generates vector representations for the training data and the testing data. The program also generates a file_names.txt file which correlates to the training/testing vector files and allows for easy tracking for the LR.py program. The same program can be run for both part a and part c, using the respective functions.

LR.py is driven by the LR function, which gets the proper files, including a special file for wrong predictions (to make investigation easy) and the learning rate. The LR function collects and randomizes the training examples (from the net vector file) and initializes the weights to all 0’s. This includes an extra slot for the b value, which I coded as simply an extra feature which is 1 for every example. The function then runs the loop which iterates through the training examples until either the loss is below the stopping value, or the function has surpassed the max iteration amount. Cross entropy and stochastic gradient descent are calculated by their respective functions. 

The program then collects test examples and predicts them using the model, outputting predictions into a file. Wrong predictions are put into a special file and sorted by how poorly the model performed. Note that both functions have a root_dir at the top (and a training_root_dir for pre-process.py) to make it easier to download and run the program.

# How to Run the Program:

To run the program, at the top of both pre-process.py and LR.py, enter the correct root file path for regular files and for the training data. I separated them to make it easier to have all of the training and testing movie reviews in a separate locatoin. You can enter the correct file paths for all your files (including the files you want to create) at the bottom of LR.py and pre-process.py, if need be, but these should not need to be changed. Make sure the file names match between pre-process.py and LR.py if you choose to change the names. You can also set the number of words you want to use in pre-process.py, and set the learning rate in LR.py.

You can download the movie review from the following link: https://ai.stanford.edu/~amaas/data/sentiment/


