import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets

from MNIST_Dataloader import MNIST_Dataloader

class NeuralNetwok: 
    def __init__(self, input_size=28*28, output_size=10, h_layers=1, h_neurons_per_layer=128):
        self.input_size = input_size
        self.output_size = output_size
        self.h_layers = h_layers
        self.h_neurons_per_layer = h_neurons_per_layer
        self.layers = self.init_layers(input_size, h_neurons_per_layer, output_size)

    # TODO: implement a programmable amount of hidden layer initialization
    def init_layers(self, input_size, h_neurons_per_layer, output_size):
        '''
        Get layer size info and develop weight array 
        initialize random weights for each connection to next layer
            weight array of output size, in array for every input node 
        return these weight arrays for each node as layer
        '''
        layer1 = np.random.uniform(-1.,1.,size=(input_size, h_neurons_per_layer))\
            /np.sqrt(input_size * h_neurons_per_layer)
        
        layer2 = np.random.uniform(-1.,1.,size=(h_neurons_per_layer, output_size))\
            /np.sqrt(h_neurons_per_layer * output_size)
        
        return [layer1, layer2]
    
    def desired_array_out(self, label):
        '''Turn label into desired output array 
        input label         5
        return desire array [0 0 0 0 0 1 0 0 0 0]
        '''
        desired_array = np.zeros(self.output_size, np.float32)
        desired_array[label] = 1
        
        return desired_array

#Sigmoid funstion
def sigmoid(x):
    return 1/(np.exp(-x)+1)    

#derivative of sigmoid
def d_sigmoid(x):
    return (np.exp(-x))/((np.exp(-x)+1)**2)

#Softmax
def softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)

#derivative of softmax
def d_softmax(x):
    exp_element=np.exp(x-x.max())
    return exp_element/np.sum(exp_element,axis=0)*(1-exp_element/np.sum(exp_element,axis=0))

    #forward and backward pass
def forward_backward_pass(x,y,nn):
    targets = np.zeros((len(y),10), np.float32)
    targets[range(targets.shape[0]),y] = 1

    l1 = nn.layers[0]
    #print(l1)
    l2 = nn.layers[1]
    # forward pass
    for i in range(0,len(x)):
        x_i=x[i]
        print(l1.shape)
        print(l1.shape[0])
        print(l1.shape[1])
        #print(x_i)
        #for layer_i in range(0,l1.shape[0]):
            #neuron = l1[layer_i]
        for neuron_i in range(0,l1.shape[1]):
            neuron = l1[:,neuron_i]
            #print(len(neuron))
            #print(len(np.array(x_i).flatten()))
            x_l1 = np.dot(neuron,np.array(x_i).flatten())
            x_sigmoid=sigmoid(x_l1)

            x_l2=x_sigmoid.dot(l2)
            out=softmax(x_l2)

            # backpropogation l2
            error=2*(out-targets)/out.shape[0]*d_softmax(x_l2)
            update_l2=x_sigmoid.T@error

            #backpropogation l1
            error=((l2).dot(error.T)).T*d_sigmoid(x_l1)
            update_l1=x.T@error

    return out,update_l1,update_l2

def get_input(x_train, y_train, idx):
    return x_train[idx], y_train[idx]

def main():
    # dataloader = MNIST_Dataloader()
    # dataloader.show_images(5, 5)
    # dataloader.simple_show()

    nn = NeuralNetwok()
    dataloader = MNIST_Dataloader()
    x_train, y_train = dataloader.get_train_data()
    '''print(len(x_train))
    images = x_train
    i = 60000
    plt.subplot(110 + 1)  # + i)
    plt.imshow(images[i], cmap=plt.get_cmap('gray'))
    plt.show()
    # dataloader.simple_show()
    print(y_train[i])'''

    #l1, _ = get_l1(x_train,y_train,0)
    #x,y = get_input(x_train,y_train,0)
    out,update_l1,update_l2=forward_backward_pass(x_train, y_train, nn)
    print('out = {}'.format(out))

    print(nn.desired_array_out([3]))
    print(nn.desired_array_out([9]))
    
    
  
if __name__=="__main__":
    main()
