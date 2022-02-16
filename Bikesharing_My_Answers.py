import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        ''' the inputs for NeuralNetwork class are : number of input nodes, number of hidden nodes, number of output nodes and 
            learning rate; this is to define the structure of the neural network        
        '''
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))
        '''matrix need to have the same number of rows as the number of input variables and
         columns has to match the number of hidden nodes'''

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        '''row count needs to match number of hidden notes; column needs to match output notes'''
        
        
        
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
      ##self.activation_function = lambda x : 1/(1+np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
                
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        def sigmoid(x):
            return  1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation here
        self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0] 
        ''' features is a matrix of explainary variables, features.shape is a list
                                          the first element is the number of records, can be called through shape[0]'''
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape) 
        ''' initialize the deltas to 0, shape is same as input to hidden weight matrix'''
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape) 
        ''' initialize the deltas to 0, shape is same as input to hidden weight matrix'''

        
        
        for X, y in zip(features, targets): 
            ''' for each record, go through the following calculations'''
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)

            
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)
        ''' after getting the delta for each record, update the weights with the delta'''


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X,self.weights_input_to_hidden) # signals into hidden layer 
        ''' hidden inputs is the dot product of X and weights [1,hidden_nodes]'''
        
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        ''' hidden output is the sigmoid of the dot product  []'''

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        ''' final inputs is the dot product of the output of the hidden layer '''
        
        final_outputs = self.activation_function(final_inputs) # signals from final output layer
        ''' final output is the sigmoid of final_inputs'''
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        '''lesson 2.4 - 2.9'''
        '''loop is at the train step, no need to loop; each step reporesents one row'''
        '''in this step, we are looking at 1 record:
           x is [1,num_inputs]
           y is [1,num_outputs] here num_output=1 so [1,1]
           weight_input_to_hidden is [num_input,1]
           weight_hidden_to_output is [num_hidden_output, num_output], here num_output=1 so [num_hidden_nodes,1]
          
        '''

        # TODO: Output error - Replace this value with your calculations.
        error = y-final_outputs # Output layer error is the difference between desired target and actual output.
        # TODO: Calculate the hidden layer's contribution to the error
        '''error is a (1,)'''

        
        '''weights_hidden_to_output is [num_hidden_nodes,1]'''

        hidden_error =  np.dot(self.weights_hidden_to_output,error)
        
        '''hidden error is (1,)  
        weights_hidden_to_output: 2x1
        hidden_error: 2x1'''
            

        
        '''hidden error is equal to the error of the output dot product by the weights FROM the hidden layer to the output
        kind of redistributing the output errors to hidden layer by the weights'''

       
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error*final_outputs*(1-final_outputs)
        '''output error term= error*f_prime(h), h is the inputs; f_prim(h) is the derivitive of the sigmoid function
           (target-prediction)*prediction*(1-prediction)
           if target=1, then it's (1-prediction)^2*prediction; if prediction is accurate, then the term is small; otherwise big
           if target=0 then it's (-prediction)^2*(1-prediction); if predication is accurate then the absolute value of the term is 
           small; 
           otherwise the absolute value is big
           
           this is a number
          
                 ''' 
    
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        ''' similar to the output error term, it is the hidden layer error * (1-output)*output
            (2,)*(2,)*(2,)=(2,)
        
        '''
            
        
        # TODO: Add Weight step (input to hidden) and Weight step (hidden to output).
        # Weight step (input to hidden)

       
        delta_weights_i_h += hidden_error_term *X[:,None]
        ''' hidden_error_term: (2,)
            X[:,None]: 3x1
            delta_weights_i_h: (2,)*(3,1)=(3,2)
            '''
        # Weight step (hidden to output)

        
        delta_weights_h_o += output_error_term *hidden_outputs[:,None]
        '''hidden output is the input to the output layer
          output_error_term: 2x1
          hidden_outputs:(2,)
          delta_weights_h_o: 2x1
        
        '''
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        
        # TODO: Update the weights with gradient descent step
        self.weights_hidden_to_output += self.lr*delta_weights_h_o/n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr*delta_weights_i_h/n_records # update input-to-hidden weights with gradient descent step
        
       
        
        '''
        update the weight with the  learning rate times the average of the weight change 
        
        '''
        
        
    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features[:,None],self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs,self.weights_hidden_to_output) # signals into final output layer
        final_outputs = self.activation_function(final_inputs) # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 10000
learning_rate = 0.05
hidden_nodes = 3
output_nodes = 1
