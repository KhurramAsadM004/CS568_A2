import numpy as np
from sklearn.preprocessing import StandardScaler
import json
import copy
import matplotlib.pyplot as plt

def normalize_data(data):   
    """
    This function calculates the mean and variance of each image in the input data and uses them to
    standardize the data. It ensures that each feature has a mean of 0 and a standard deviation of 1.
    """
    eps=1e-8
    mean = np.mean(data, axis=1, keepdims=True)
    variance = np.var(data, axis=1, keepdims=True)
    data_norm = np.divide((data - mean), np.sqrt(variance+eps))
    return data_norm

def sigmoid(a):
    """
    Computes the sigmoid function element-wise on 'a'.  
    'a' is the pre-activation value of the neuron.
    'z' is the output after applying the sigmoid activation function.    
    """
    z =  1/(1 + np.exp(-a))
    return z

def tanh(a):
    """
    Computes hyperbolic tangent (tanh) function element-wise on 'a'.  
    'a' is the pre-activation value of the neuron.
    'z' is the output after applying the tanh activation function.    
    """    
    z = (np.exp(a) - np.exp(-a)) / (np.exp(a) + np.exp(-a))
    return z

def relu(a):
    """
    Computes the rectified linear unit (ReLU) function element-wise on 'a'.    
    'a' is the pre-activation value of the neuron.
    'z' is the output after applying the relu activation function.    
    """
    z = np.where(a > 0, a, 0)
    return z
  

def lrelu(a, k=0.01):
    """
    Computes the leaky rectified linear unit (Leaky ReLU) function element-wise on 'a' with a given slope 'k'.
    'a' is the pre-activation value of the neuron.
    'z' is the output after applying the lrelu activation function.    
    """
    z = np.where(a > 0, a, a * k)
    return z

def identity(a):
    return a

def sigmoid_derivative(z):
    """
    Computes the derivative of the sigmoid function element-wise on 'z'.   
    'hprime' contains sigmoid function derivative values.    
    """
    hprime = z * (1-z)
    return hprime

def tanh_derivative(z):
    """
    Computes the derivative of the hyperbolic tangent (tanh) function element-wise on 'z'. 
    'hprime' contains tanh function derivative values. 
    """
    hprime = 1 - (z**2)
    return hprime

def relu_derivative(z): 
    """
    Computes the derivative of the rectified linear unit (ReLU) function element-wise on 'z'.
    'hprime' contains relu function derivative values. 
    """
    hprime = np.where(z > 0, 1, 0)
    return hprime

def lrelu_derivative(z, k=0.01): 
    """
    Computes the derivative of the leaky rectified linear unit (Leaky ReLU) function element-wise on 'z'
    with a given slope 'k'.   
    'hprime' contains lrelu function derivative values. 
    """    
    hprime = np.where(z > 0, 1, k)
    return hprime

def identity_derivative(z):
    return np.ones_like(z)


def softmax(a):
    max_a = np.max(a) #compute maximum of pre-activations in a (1 max per sample if using mini-batches)
    a_exp = np.exp(a - max_a) #exponentials of pre-activations after subtracting maximum
    a_sum = sum(a_exp) #sum of exponentials
    z = np.divide(a_exp, a_sum) #softmax results [max(i,eps) for i in a_sum]
    return z


def mse(self, y, batch_target):
    loss = np.sum(np.square(y - batch_target)) / len(y) #mean squared loss for regression
    return loss

def mce(self, y, batch_target):
    # print("Y:", y.shape)
    # print("T: ", batch_target.shape)
    eps = 1e-9

    loss = 0
    m = 0
    n = 0
    for i in y:
        n = 0
        y_loss = 0
        for j in i:
            log_value = np.log(max(j, eps))
            y_loss += batch_target[m][n] * log_value
            n += 1 
        loss += y_loss
        m += 1
    loss = -loss #mean multiclass cross-entropy loss for multiclass classification
    # print(len(batch_target[0]))
    return loss / len(batch_target[0])

def bce(self, y, batch_target):
    eps = 1e-9
    
    y_pred = np.clip(y, eps, 1 - eps)
    y_actual = batch_target

    first_part = y_actual * np.log(y_pred + eps)    
    second_part = (1-y_actual) * np.log(1-y_pred + eps)
    
    loss = -np.mean(first_part+second_part, axis=0) #mean binary cross-entropy loss for binary classification
    
    return loss

def save_model_details(self):
    model_details = {
    'layer_dim': self.num_neurons,
    'activations': self.activations_func,
    'optimizer': self.optimizer,
    'epochs': self.epochs,
    'loss': self.loss,
    'batch_size': self.mini_batch_size,
    'learning_rate': self.learning_rate,
    'mode': self.mode,
    'weights_save_dir': self.weights_save_dir    
    }
    # open a file and use the json.dump method to save model details 
    filename = self.weights_save_dir+'model_details.json'
    with open(filename, "w") as json_file:
        json.dump(model_details, json_file)

    # save model weights as a numpy array in a .npy file
    filename = self.weights_save_dir+'model_weights.npy'
    np.save(filename, self.parameters)
    
        
def load_model_details(self):
    with open(self.weights_save_dir+'model_details.json', 'r') as json_file:
        model_details = json.load(json_file)
        
        self.num_neurons = model_details['layer_dim']
        self.activations_func = model_details['activations']
        self.optimizer = model_details['optimizer']
        self.epochs = model_details['epochs']
        self.mini_batch_size = model_details['batch_size']
        self.learning_rate = model_details['learning_rate']
        self.mode = model_details['mode']
        # '''ADD CODE HERE''' # similarly for other details (HINT: see save_model_details() function above)
        self.weights_save_dir = model_details['weights_save_dir']
        print(model_details)
        
    loaded_weights = np.load(self.weights_save_dir+'model_weights.npy', allow_pickle=True)

    # The .item() method is used here to convert the numpy array into a dictionary-like structure
    # that was originally used to save and store model parameters.
    self.parameters = loaded_weights.item() 
    # print(self.parameters)



def plot_loss(self, loss, val_loss):        
    """
    Plot the training and validation loss curves and save figure    
    """
    plt.figure() 
    fig = plt.gcf() 
    plt.plot(loss, linewidth=3, label="train") # Plot the training loss with a line width of 3 and label="train"
    plt.plot(val_loss, linewidth=3, label="val")  # Plot the validation loss with a line width of 3 and label="val"
    plt.ylabel('Error Function')
    plt.xlabel('Epochs')
    plt.title('learning rate =%s' % self.learning_rate) # Set the title of the plot 
    plt.grid() 
    plt.legend()
    plt.show() 
    fig.savefig('plot_loss.png')  
    plt.close()
    
        
def plot_weights(self):
    """
    Plot the average magnitude of weights for each layer and save figure
    """
    avg_l_w = [] # Create an empty list to store the average weights for each layer
    param = copy.deepcopy(self.parameters)  # Create a deep copy of the network parameters
    for l in range(1, self.num_layers+1): 
#             print("layer %s"%l)
         weights = param['W%s' % l]  # Get the weights for the current layer
         dim = weights.shape[0]  # Get the number of weights in the current layer
         avg_w = []  # Create an empty list to store the average weight magnitude 
         for d in range(dim):  
             abs_w = np.abs(weights[d]) # Calculate the absolute value of each weight 
             avg_w.append(np.mean(abs_w))         
         temp = np.mean(avg_w)  
         avg_l_w.append(temp) 
    layers = ['L %s'%l for l in range(self.num_layers+1)]  # Create a list of layer names for the x-axis labels
    weight_mag = avg_l_w  # Magnitudes of average weights for each layer
    plt.xticks(range(len(layers)), layers)  
    fig = plt.gcf() 
    plt.xlabel('layers') 
    plt.ylabel('average weights magnitude')
    watermark_text = 'Official Solution'
    plt.text(0.5, 0.5, watermark_text, fontsize=20, color='gray', ha='center', va='center', alpha=0.5, transform=plt.gca().transAxes)
    plt.title('')  # Set the title of the plot
    plt.bar(range(len(weight_mag)),weight_mag, color='blue', width=0.2)  # Create a bar plot of average weights magnitude for each layer
    plt.show()  
    fig.savefig(self.function_name+'_plot_weights.png') 
    plt.close(fig)
   
   
def plot_gradients(self):
    """
    Plot the average magnitude of weights' gradients for each layer and save figure
    """
    avg_l_g = [] # list to store layer averages
    # Iterate over each layer
    for l in range(1, self.num_layers+1):
        #print("layer %s"%l)
        weights_grad = self.grads['dW%s' % l]   # gradients for the weights layer l
        dim = weights_grad.shape[0]             # number of neurons in layer l
        avg_g = []                              # list to store neuron averages
        # Iterate over each weight in the current layer
        for d in range(dim):
            # average of absolute gradients of weights of neuron d
            avg_g.append(np.mean(np.abs(weights_grad[d])))
        # average of absolute gradients of layer l
        avg_l_g.append(np.mean(avg_g))

    layers = ['L %s'%l for l in range(self.num_layers+1)]   # Create a list of layer names for the x-axis labels
    weights_grad_mag = avg_l_g                              # Magnitudes of average gradients for each layer
    fig = plt.gcf()                                         # Get the current figure
    plt.xticks(range(len(layers)), layers)                  # Set the x-axis tick labels
    plt.xlabel('layers')                                    # Set the x-axis label
    plt.ylabel('average gradients magnitude')               # Set the y-axis label
    watermark_text = 'Official Solution'
    plt.text(0.5, 0.5, watermark_text, fontsize=20, color='gray', ha='center', va='center', alpha=0.5, transform=plt.gca().transAxes)
    #plt.ylim(0, 0.5)
    plt.title('activation function: {}'.format(self.activations_func[1]))           # Set the title of the plot
    plt.bar(range(len(weights_grad_mag)),weights_grad_mag, color='red', width=0.2)  # Create a bar plot of average gradients magnitude for each layer
    plt.show()                                  # Display the plot   
    fig.savefig(self.function_name+'_plot_gradients.png')      # Save the figure as a PNG image
    plt.close(fig)
