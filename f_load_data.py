from f_utils import *
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io


def load_dataset(args):
    print("loading data.....")
    
    data_file = scipy.io.loadmat(args.data_dir+'mnist_uint8.mat')
    
    train_x = data_file['train_x']
    train_y = data_file['train_y']
    test_x = data_file['test_x']
    test_y = data_file['test_y']

#    print(train_y[0:20], train_y[1000:1020])
#    df
    print("train_x, train_y, test_x, test_y",train_x.shape, train_y.shape, test_x.shape, test_y.shape)
    
#    class1=7
#    class2=1
#    
#     #training data
#    class1_inds=np.where(train_y[:,class1]==1)
#    class2_inds=np.where(train_y[:,class2]==1)
#    class11_inds = np.asarray(class1_inds)
#    class22_inds = np.asarray(class2_inds)
#    
#      
#    n1=class11_inds.shape[1]
#    n2=class22_inds.shape[1]
#    n=n1+n2
#
#    n1=class11_inds.shape[1]
#    n2=class22_inds.shape[1]
#    n=n1+n2
#    
#    training_inds=np.hstack([class1_inds, class2_inds])
#    train_x=train_x[training_inds,:]
#    
#    train_x = np.reshape(train_x,[train_x.shape[1],train_x.shape[2]])
#    train_t = np.zeros((n,1))
#    train_t[0:n1,:] = 1
#    
#    
#    class1_inds=np.where(test_y[:,class1]==1)
#    class2_inds=np.where(test_y[:,class2]==1)
#    class11_inds = np.asarray(class1_inds)
#    class22_inds = np.asarray(class2_inds)
#    
#    n1=class11_inds.shape[1]
#    n2=class22_inds.shape[1]
#    testn=n1+n2
#    testing_inds=np.hstack([class1_inds , class2_inds])
#    test_x = test_x[testing_inds,:]
#    test_t = np.zeros((testn,1))
#    test_t[0:n1,:] = 1
#    test_x = np.reshape(test_x,[test_x.shape[1],test_x.shape[2]])
    
    train_x, val_x, train_t, val_t = train_test_split(train_x, train_y, test_size=0.2)

    train_x = normalize_data(train_x)
    val_x = normalize_data(val_x)
    test_x = normalize_data(test_x)   
    
    
    
    return train_x.T, train_t.T, val_x.T, val_t.T, test_x.T, test_y.T   
