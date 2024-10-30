#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:04:01 2023

@author: reichert
"""
import numpy as np
from numpy.random import randn
from numpy.linalg import norm

class DenseLayer:
    """
    classe DenseLayer
    """
    def __init__(self, layer_dim, activation):
        acts = ['relu','sigmoid','linear','gaussian','softmax']
        self.layer_dim = layer_dim
        if activation in acts:
            self.activation = activation
        else:
            raise ValueError('Unknown activation '+activation)
        self.A = np.array([]) # initialiser A avec un array vide
        self.Z = np.array([]) # pareil
        self.parameters = {} # dictionary pour les paramètres
        self.grads = {} # dictionary pour les gradients
        
        
class NNmodel:
    """ 
    classe NNmodel

    """
    def __init__(self, input_dim):
        self.input_dim = input_dim # dimension des données d'entrée
        self.layers = [None] # list vide pour les couches
        self.loss = '' # string pour le loss
        
    def add_layer(self, layer):
        """
        fonction add_layer
        """
        self.layers.append(layer)
        
    def initialize(self, loss):
        """
        fonction initialize
        """
        np.random.seed(1)
        if not loss in ['cross_entropy', 'least_squares']:
            raise ValueError('Unknown loss' + loss)
        else:
            self.loss = loss
        L = len(self.layers)
        if L == 1:
            raise ValueError('No layers with parameters !')
        else:
            # initialize parameters of first hidden layer
            l_dim = self.layers[1].layer_dim
            W_shape = (l_dim,self.input_dim)
            W = randn(*W_shape) * np.sqrt(2/l_dim)
            b = np.zeros((l_dim,1))
            parameters = self.layers[1].parameters
            parameters['W'] = W
            parameters['b'] = b
            # initialize remaining layers
            if L > 2:
                for l in range(2,L):
                    l_dim = self.layers[l].layer_dim
                    l_dim_m = self.layers[l-1].layer_dim
                    W_shape = (l_dim,l_dim_m)
                    W = randn(*W_shape) * np.sqrt(2/l_dim_m)
                    b = np.zeros((l_dim,1))
                    parameters = self.layers[l].parameters
                    parameters['W'] = W
                    parameters['b'] = b            
        
    def forward_prop(self, X):
        """
        fonction forward_prop
        """
        layers = self.layers
        L = len(layers)
        Am = X
        for l in np.arange(1,L):
            W = layers[l].parameters['W']
            b = layers[l].parameters['b']
            act = layers[l].activation
            Z = np.dot(W, Am) + b
            layers[l].Z = Z
            if act == 'relu':
                layers[l].A = relu(Z)
            elif act == 'sigmoid':
                layers[l].A = sigmoid(Z)
            elif act == 'linear': 
                layers[l].A = linear(Z)
            elif act == 'gaussian':
                layers[l].A = np.zeros(Z.shape)
            elif act == 'softmax':
                layers[l].A = softmax(Z)

            Am = layers[l].A
            
    def back_prop(self, X, Y):
        """
        fonction back_prop
        """
        loss = self.loss
        layers = self.layers
        L = len(layers)
        if L == 1:
            raise ValueError('No layers with parameters !')
        m = X.shape[1]
        
        # initialisation
        AL = layers[-1].A
        ZL = layers[-1].Z
        if L > 2:
            Am = layers[-2].A
        else:
            Am = X
            
        # gradient dernière couche
        if loss == 'cross_entropy':
            if layers[-1].activation == 'sigmoid':
                deltaL =  cross_entropy_grad(AL, Y)  \
                         *sigmoid_prime(ZL) # BP1
            elif layers[-1].activation == 'softmax':
                deltaL =  cross_entropy_grad(AL, Y)  \
                         *softmax_prime(ZL) # BP1

            
            else:
                raise ValueError('Cross entropy cost expects sigmoid output')
            
        if loss == 'least_squares':
            if layers[-1].activation =='sigmoid':
                deltaL =  least_squares_grad(AL, Y)  \
                         *sigmoid_prime(ZL) # BP1
            elif layers[-1].activation == 'relu':
                deltaL =  least_squares_grad(AL, Y)  \
                         *relu_prime(ZL) # BP1
            elif layers[-1].activation == 'linear':
                deltaL =  least_squares_grad(AL, Y)  \
                         *linear_prime(ZL) # BP1
            elif layers[-1].activation == 'gaussian':
                deltaL =  least_squares_grad(AL, Y)  \
                         *gaussian_prime(ZL) # BP1
            
        db = 1/m * np.sum(deltaL, axis=1, keepdims=True) # BP3
        dW = 1/m * np.dot(deltaL,Am.T) # BP4
        layers[-1].grads = {'dW': dW, 'db': db}
        
        # boucle pour gradients des couches cachées de L-1 à 2
        if L > 2:
            deltaP = deltaL
            for l in np.arange(2, L-1):
                #print(l)
                #Al = A[-l]
                Am = layers[-l-1].A
                Zl = layers[-l].Z
                Wp = layers[-l+1].parameters['W']
                if layers[-l].activation == 'relu':
                    delta = np.dot(Wp.T, deltaP) * relu_prime(Zl)
                elif layers[-l].activation == 'sigmoid':
                        delta = np.dot(Wp.T, deltaP) * sigmoid_prime(Zl)
                elif layers[-l].activation == 'gaussian':
                        delta = np.dot(Wp.T, deltaP) * gaussian_prime(Zl)
                        
                
                db = 1/m * np.sum(delta, axis=1, keepdims=True)
                dW = 1/m * (np.dot(delta, Am.T)) 
                layers[-l].grads = {'dW': dW, 'db': db} 
                deltaP = delta
    
            #première couche cachée
            Z1 = layers[1].Z
            Am = X
            Wp = layers[2].parameters['W']
            if layers[1].activation == 'relu':
                delta1 = np.dot(Wp.T, deltaP) * relu_prime(Z1)
            elif layers[1].activation == 'sigmoid':
                    delta1 = np.dot(Wp.T, deltaP) * sigmoid_prime(Z1)
            elif layers[1].activation == 'gaussian':
                    delta1 = np.dot(Wp.T, deltaP) * gaussian_prime(Z1)
                
            db = 1/m * np.sum(delta1, axis=1, keepdims=True) # BP3
            dW = 1/m * np.dot(delta1,Am.T) # BP4
            layers[1].grads = {'dW': dW, 'db': db}
        
    def train(self, database, num_iterations, learning_rate):
        """ 
        fonction train
        """
        X = database['X_train']
        Y = database['Y_train']
        costs = np.array([])
        if 'X_valid' in database.keys():
            X_valid = database['X_valid']
            Y_valid = database['Y_valid']
            val_costs = np.array([])
            
        layers = self.layers
        loss = self.loss
        if len(layers) == 1:
            raise ValueError('No trainable layers !')
            
        #-- Gradient descent
        for i in np.arange(num_iterations):

            # propagation avant et coût validation
            if 'X_valid' in database.keys():
                self.forward_prop(X_valid)
                AL = layers[-1].A
                if loss == 'cross_entropy':
                    val_cost = cross_entropy_cost(AL, Y_valid)
                elif loss == 'least_squares':
                    val_cost = least_squares_cost(AL, Y_valid)          

            # propagation avant train
            self.forward_prop(X) 

            # calculer coût
            AL = layers[-1].A
            if loss == 'cross_entropy':
                cost = cross_entropy_cost(AL, Y)
            elif loss == 'least_squares':
                cost = least_squares_cost(AL, Y)
                
            # rétropropagation
            self.back_prop(X, Y)

            # mise à jour des paramètres
            for layer in layers[1:]:
                layer.parameters['W'] -= learning_rate*layer.grads['dW']
                layer.parameters['b'] -= learning_rate*layer.grads['db']
            #-- Record the costs
            costs = np.append(costs, cost)
            if 'X_valid' in database.keys():
                val_costs = np.append(val_costs, val_cost)
            
        #final return statement
        if 'X_valid' in database.keys():
            return costs, val_costs
        else:
            return costs
    
    def predict(self, X):
        """ 
        fonction predict
        """
        loss  = self.loss
        m = X.shape[1]
      
        if loss == 'cross_entropy':
            y_prediction = np.zeros((1,m))
            # propagation avant
            self.forward_prop(X)
            probas = self.layers[-1].A
            M = probas.shape[0]
            if M==1:
                for i in np.arange(m):
                    if probas[0,i] >= 0.5:
                        y_prediction[0,i] = 1
                    else:
                        y_prediction[0,i] = 0
            else:
                for i in np.arange(m):
                    for k in np.arange(M): 
                        if probas[k,i] >= np.max(probas[:,i]):
                            y_prediction[0,i] = k
        
        
        elif loss == 'least_squares':
            self.forward_prop(X)
            y_prediction = self.layers[-1].A
            
        return y_prediction
    
    def eval_nn(self, X):
        """ 
        fonction eval
        """
        self.forward_prop(X)
        y_prediction = self.layers[-1].A
        return y_prediction
    
                
# fonctions auxiliaires
def relu(Z):
    """
    relu function
    """
    return np.maximum(Z,0)

def sigmoid(Z):
    """
    sigmoid function
    """
    return 1./(1+np.exp(-Z))

def linear(Z):
    return Z

def softmax(Z):
    res = np.exp(Z)
    eps = np.finfo(float).eps
    res = res/np.sum(res+eps, axis = 0)
    return res

def gaussian(Z):
    return Z

def relu_prime(Z):
    """
    relu derivative
    """
    res = np.ones_like(Z)
    res[Z <= 0] = 0
    return res

def sigmoid_prime(Z):
    """
    sigmoid derivative
    """
    return sigmoid(Z)*(1 - sigmoid(Z))

def softmax_prime(Z):
    
    return softmax(Z)*(1 - softmax(Z))


def linear_prime(Z):
    return np.ones(Z.shape)


def gaussian_prime(Z):
    return np.ones(Z.shape)

def cross_entropy_cost(AL, Y):
    """ 
    cost function
    """
    m = Y.shape[1]
    #-- Compute the cross-entropy cost
    eps = np.finfo(float).eps

    
    cost = -( np.sum(Y*np.log(AL+eps)) + np.sum((1-Y)*np.log(1-AL+eps)))/ m  
    print(cost)
    #cost = -( np.dot(Y, np.log(AL+eps).T) + np.dot(1-Y, np.log(1-AL+eps).T) ) / m  
    cost_val = cost
    
    return cost_val

def least_squares_cost(AL, Y):
    m = Y.shape[1]
    cost =  0
    
    return cost

def cross_entropy_grad(AL,Y):
    eps = np.finfo(float).eps # epsilon machine
    
    return - (Y/(AL+eps)) + (1-Y)/(1-AL + eps)
        
def least_squares_grad(AL,Y):
    return np.zeros(AL.shape)    
        
        
        
        