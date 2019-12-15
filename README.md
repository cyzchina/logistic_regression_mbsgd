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

python3.6 build:  
cd python  
source ve_python3.6/bin/activate  
python setup.py build  
deactivate  
ln -s build/lib.linux-x86_64-3.6/mbsgd.cpython-36m-x86_64-linux-gnu.so mbsgd.so  

python3.6 install:  
cd python  
source ve_python3.6/bin/activate  
python setup.py bdist_wheel  
deactivate  
pip3.6 install dist/mbsgd-2.0-cp36-cp36m-linux_x86_64.whl  

python3.6 example:  
import pandas as pd  
from mbsgd import MBSGD  

train_data = pd.read_csv('train.csv')  
test_data = pd.read_csv('test.csv')  

new_data = test_data.iloc[:, 1:]  
test_labels = test_data.iloc[:, 0]  

# params train_data, <maxit>, <eps>, <cpus>, <alpha>, <l1>, <gama>  
trainer = MBSGD(train_data, cpus=4)  
# params  <maxit>, <eps>, <cpus>, <alpha>, <l1>, <gama>  
model = trainer.Train(maxit=4)  
print(model.cost_sec)  

# Predict(new_data, 1) produces classification result  
# Predict(new_data) produces regression result  
result = model.Predict(new_data, 1)  

from sklearn.metrics import roc_auc_score  
auc_score = roc_auc_score(test_labels, result)  
print(auc_score)  
