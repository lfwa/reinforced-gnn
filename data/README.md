## Data
This directory contains the data used for training and testing our models. The data consists of `10000 users` and `1000 items` with user-item ratings between 1 and 5.

## Structure of Directory
- `data_test.csv`: This file contains the test data used for evaluation. The file contains a sample submission format, where the matrix entries which we need to predict are specified in this file. The predicted rating of this file is set to `3` as a dummy value.
- `data_train.csv`: This file contains the training data used for our models. The file contains on each row an ID corresponding to some row and column of the user-item matrix together with an observed rating for this entry.
