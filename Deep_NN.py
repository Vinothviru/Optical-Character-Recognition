import numpy as np
import math as m 
"""
Problem description:To identify the characters.
"""
class Deep_NN:
	def __init__(self,no_input_features,no_inputs,no_hidden_nodes,no_hidden_layers,no_output_nodes):
		"""
		Initialization:Here the initialized variables will be used in the forward propogation.
		"""
		"""
		After getting the input from user the input vector,input weights and input bias are initialized.
		"""
		self.no_input_features=no_input_features
		self.no_inputs=no_inputs
		self.input_vector=np.zeros((self.no_input_features,self.no_inputs))
		self.input_weights=np.random.randn(no_hidden_nodes,self.no_input_features)*0.01
		self.input_bias=np.random.rand(no_hidden_nodes,1)
		"""
		After getting the input from user the hidden layers,hidden weights and hidden bias are constructed.
		"""	
		self.no_hidden_nodes=no_hidden_nodes
		self.no_hidden_layers=no_hidden_layers
		self.hidden_layer=np.zeros((self.no_hidden_layers,self.no_hidden_nodes,1))
		if (self.no_hidden_layers>1):
			self.hidden_weights=np.random.randn(self.no_hidden_layers-1,self.no_hidden_nodes,self.no_hidden_nodes)
			self.hidden_bias=np.random.randn(self.no_hidden_nodes,1)
		"""
		After getting the input from user the output vector is constructed.
		"""			
		self.no_output_nodes=no_output_nodes
		self.output_layer=np.zeros((self.no_output_nodes,1))
		self.output_weights=np.random.randn(self.no_output_nodes,self.no_hidden_nodes)*0.01
		self.output_bias=np.array([[0]])
		"""
		Thus the initialization part gets over 
		"""
	
	def compute_hidden_z(self,input_vector):
		"""
		Computing the Z value of the hidden 1st layer by getting the input vector as the input 
		"""
		self.hidden_z_value=(self.input_weights.dot(input_vector))+self.input_bias
		return self.hidden_z_value
	
	def tanhh(self,compute_hidden_z):
		"""
		Computing the RElu function for hidden 1st hidden layer using the formula g(z)=(exp(z)-exp(-z))/(exp(z)+exp(-z))
		"""
		#print("ghhghg")
		self.tanhhh=(np.exp(compute_hidden_z)-np.exp(-(compute_hidden_z)))/(np.exp(compute_hidden_z)+np.exp(-(compute_hidden_z)))
		#print("ghhghghjdydxc")
		return self.tanhhh
	
	def reLU(self,compute_hidden_z):
		self.reLUU=np.maximum(0,compute_hidden_z)
		return self.reLUU
	
	def compute_output_z(self,hidden_layer):
		"""
		Computing the Z value of the output layer by getting the hidden_layer value as the input
		"""
		self.output_z_value=(self.output_weights.dot(hidden_layer))+self.output_bias
		return self.output_z_value
	
	def sigmoid(self,output_z):
		"""
		Computing the Sigmoid function for output layer using the formula g(z)=1/(1+(exp^-z))
		"""
		self.predicted_value=1/(1+np.exp(-(output_z)))
		return self.predicted_value
		
	def Forward_propogation(self,input_vector):
		"""
		Here the first process of Neural Network is computed that is  Forward propogation method
		"""	
		#print("dudo")
		self.input_vector=input_vector
		self.hidden_layer_z=self.compute_hidden_z(self.input_vector)
		#print(self.hidden_layer_z.shape)
		self.hidden_layer=self.reLU(self.hidden_layer_z)#self.tanhh(self.hidden_layer_z)#self.sigmoid(self.hidden_layer_z)#self.tanhh(self.hidden_layer_z)#self.reLU(self.hidden_layer_z)
		#print(self.hidden_layer.shape)
		self.output_layer_z=self.compute_output_z(self.hidden_layer)
		#print(self.output_layer_z.shape)
		self.output_layer=self.sigmoid(self.output_layer_z)
		#print(self.output_layer_z.shape)
		#print(self.output_layer)
		return self.output_layer
		"""
		test=Deep_NN(3,4,1,1)
		input_vector=np.array([[1],[2],[3]])#np.array([[[1,2,3,4,5,6,7],[10,20,30,40,50,60,70],[100,200,300,400,500,600,700]]])
		test.Forward_propogation(input_vector[:])	
		"""	
	
	def Back_propogation(self,actual_output_vector):
		"""
		Here the coputation of backporopagation is done.
		Thus the corrected values of input weights,bias and output weights,bias are identified by knowing the output error and hidden error. 
		"""
		self.actual_output_vector=actual_output_vector
		self.output_error=self.output_layer-self.actual_output_vector
		self.derivative_output_weight=(1/self.no_inputs)*(self.output_error.dot(self.hidden_layer.T))
		self.derivative_output_bias=(1/self.no_inputs)*(np.sum(self.output_error,axis=1,keepdims=True))
		self.hidden_error=(self.output_weights.T.dot(self.output_error))*(1-(self.hidden_layer_z*self.hidden_layer_z))
		self.derivative_hidden_weight=(1/self.no_inputs)*(self.hidden_error.dot(self.input_vector.T))
		self.derivative_hidden_bias=(1/self.no_inputs)*(np.sum(self.hidden_error,axis=1,keepdims=True))
		return self.derivative_output_weight,self.derivative_output_bias,self.derivative_hidden_weight,self.derivative_hidden_bias
			
	def Loss_error_function(self,actual_output_vector):
		"""
		Loss function is used to find the loss which is used in the cost function 
		"""
		self.loss_value=-((actual_output_vector*(np.log(self.output_layer)))+((1-actual_output_vector)*(np.log(1-self.output_layer))))
		return self.loss_value
	
	def Cost_function(self,actual_output_vector):
		"""
		Cost of the weight and bias are identified here.thereby it should be minimized using gradient descent 
		"""
		self.cost_values=np.sum(self.Loss_error_function(actual_output_vector))
		self.final_cost=(1/self.no_inputs)*self.cost_values
		return self.final_cost
	
	def Gradient_descent(self,learning_rate,input_vector,actual_output_vector):	
		"""
		Gradient descent is used to minimize the cost function .
		That is the efficient values of weights and bias can be identified here. 
		"""
		for j in range(5000):
			#calling forward propogation
			self.call_forward=self.Forward_propogation(input_vector)
			#print("fwd dude : ",self.call_forward)
			"""Have to implement cost function here"""
			self.cost_measure=self.Cost_function(actual_output_vector)
			print("COST REDUCTION : ",self.cost_measure)
			"""calling backward Propagation"""
			self.call_backprop=self.Back_propogation(actual_output_vector)
			#print("back buddy : ",self.call_backprop)
			self.input_weights=self.input_weights-((learning_rate)*(self.derivative_hidden_weight))
			self.input_bias=self.input_bias-((learning_rate)*(self.derivative_hidden_bias))
			self.output_weights=self.output_weights-((learning_rate)*(self.derivative_output_weight))
			self.output_bias=self.output_bias-((learning_rate)*(self.derivative_output_bias))
			#print("w1 : ",self.input_weights)
			#print("b1 : ",self.input_bias)
			#print("w2 : ",self.output_weights)
			#print("b2 : ",self.output_bias)
		
		
"""test=Deep_NN(3,4,1,1,1)
input_vector=np.array([[1],[2],[3]])#np.array([[[1,2,3,4,5,6,7],[10,20,30,40,50,60,70],[100,200,300,400,500,600,700]]])
output_vector=np.array([[0],[1],[0]])	
learning_rate=0.01
test.Gradient_descent(learning_rate,input_vector,output_vector)"""

