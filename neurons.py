import numpy as np

class Neuron():
    def __init__(self,n_layers,input_size,output_size):
        self.n_layers = n_layers
        if not isinstance(input_size,tuple):
            input_size = (input_size,)
        self.input_size = input_size
        self.output_size = output_size
        self._create_weight_layers()
    
    def load_weights(self,weights):
        index = 0
        weights = self._reshape_input_weights(weights)
        for matrix_index in range(len(self.weights)):
            self.weights[matrix_index] = weights[matrix_index]
                
    def _reshape_input_weights(self,weights):
        input_weight = []
        index = 0
        for matrix_index in range(len(self.weights)):
            rows,columns = self.weights[matrix_index].shape
            size = rows*columns
            input_weight.append(weights[index:size+index].reshape(rows,columns))
            index=size+index
        input_weight = np.array(input_weight,dtype=object)
        self._check_weight_input_shape(input_weight)
        return input_weight
            
    def _check_weight_input_shape(self,weights):
        if weights.shape != self.weights.shape:
            input_weight = weights.shape
            neuron_weight = self.weights.shape
            raise ValueError(f'Las dimensiones de los pesos no coinciden (input: {input_weight}, matriz original: {neuron_weight})')
        
        total_weights = len(weights)
        for index in range(total_weights):
            if weights[index].shape != self.weights[index].shape:
                input_weight = weights[index].shape
                neuron_weight = self.weights[index].shape
                raise ValueError(f'Las dimensiones de los pesos de la {index} matriz no coinciden (input: {input_weight}, matriz original: {neuron_weight})')
        
    def _sigmoid(self,value):
        return 1/(1+np.exp(-value.astype(float)))
    
    def _create_weight_layers(self):
        self.weights = []
        self.inputs = []
        prev_input = self.input_size[0]
        for n in range(self.n_layers):
            self.weights.append(np.random.randint(-100,100,size=(self.input_size[n],prev_input))/100)
            self.inputs.append(np.zeros(shape=(prev_input,1)))
            prev_input = self.input_size[n]
            
        self.inputs.append(np.zeros(shape=(prev_input,1)))
        self.weights.append(np.random.randint(-100,100,size=(1,prev_input))/100)
        self.weights = np.array(self.weights,dtype=object)
        self.inputs = np.array(self.inputs,dtype=object)
        
    def feed_forward(self,inputs):
        if isinstance(inputs,(tuple,list)):
            inputs = np.array(inputs)
        if inputs.shape != (len(inputs),1):
            inputs = inputs.reshape(-1,1)
        self.inputs[0] = inputs
        for n_input in range(len(self.inputs)-1):
            matmul_output = np.matmul(self.weights[n_input],self.inputs[n_input]) #Z_i
            self.inputs[n_input+1] = self._sigmoid(matmul_output)
            
        output = self._sigmoid(np.matmul(self.weights[-1],self.inputs[-1]))
        return output