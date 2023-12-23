import numpy as np
from neural_network import *
from f_load_data import *
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')
parser.add_argument('--layer_dim', dest='layer_dim', type=int, nargs='+', default=[784, 50, 20, 10], help='no of layers with neurons')
parser.add_argument('--activations', dest='activations', type=str, nargs='+', default=[None, 'relu', 'relu', 'softmax'], help='activation function')
parser.add_argument('--optimizer', dest='optimizer', default='adam', choices=['sgd', 'adam'], help='optimization algorithm')
parser.add_argument('--epochs', dest='epochs', type=int, default=1000, help='epochs')
parser.add_argument('--loss', dest='loss', default='mce', choices=['mce'], help='loss functions')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='batch_size: 48, 64, 128,..')
parser.add_argument('--check_grad', dest='check_grad', action='store_true', default=False, help='check gradient ..')
parser.add_argument('--early_stopping', dest='early_stopping', action='store_true', default=False, help='Enable early stopping. If set True, training will stop when validation loss stops improving for a certain number of epochs.')
parser.add_argument('--learning_rate', dest='learning_rate', type=float, default=1e-3, help='learning_rate')
parser.add_argument('--delta', dest='delta', type=float, default=0.9, help='delta')
parser.add_argument('--gamma', dest='gamma', type=float, default=0.999, help='gamma')
parser.add_argument('--eps_for_v', dest='eps_for_v', type=float, default=1e-8, help='eps for vhat')
parser.add_argument('--patience', dest='patience', type=int, default=10, help='Number of epochs with no improvement after which training will be stopped.')
parser.add_argument('--convergence_threshold', dest='convergence_threshold', type=float, default=1e-5, help='')
parser.add_argument('--mode', dest='mode', type=str, default='test', choices=['train', 'test'], help='')
parser.add_argument('--weights_save_dir', dest='weights_save_dir', type=str, default='classifier/', help='path')
parser.add_argument('--data_dir', dest='data_dir', type=str, default='Data/', help='')
args = parser.parse_args()


nn = NeuralNetwork(args)
# load dataset
train_x, train_t, val_x, val_t, test_x, test_t = load_dataset(args)
print("train_x and train_t: ", train_x.shape, train_t.shape)
print("val_x and val_t: ", val_x.shape, val_t.shape)
print("test_x and test_t: ", test_x.shape, test_t.shape)

if args.mode == 'train':
    nn.train(train_x, train_t, val_x, val_t)
elif args.mode == 'test':
    train_acc, _ = nn.test(train_x, train_t)
    test_acc, confusion_matrix = nn.test(test_x, test_t) 
    print("training acc..", np.round(train_acc,2))
    print("testing acc..", np.round(test_acc,2))
    print("confusion_matrix: \n", confusion_matrix)

    plt.title(f'Confusion matrix | Training acc. {np.round(train_acc,2)} | Testing acc. {np.round(test_acc, 2)}') # Set the title of the plot 
    classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    sb.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.draw()
    plt.pause(0.01)
    plt.savefig('ConfusionMatrixTest.png')  
    plt.clf()

else:
    print("Please select a valid option: 'train' or 'test'.")




