# lr_mbsgd
logistic regression mini batch stochastic gradient descend  

Usage:  
./lr_mbsgd [options] [training_data]  

Options:  
-s <int>   Shuffle dataset after each iteration. default 1  
-i <int>   Maximum iterations. default 500  
-e <float> Convergence rate. default 0.005  
-a <float> Learning rate. default 0.001  
-l <float> L1 regularization weight. default 0.001  
-t <file>  Test file to classify  
-r <float> Randomise weights between -1 and 1, otherwise 0  
-c <int>   cpu cores. default 4  

Example:  
./lr_mbsgd -i 20 -c 8 -t test.csv train.csv  

csv:  
the first column is label without header and index  
