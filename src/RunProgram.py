# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 14:27:33 2016

@author: Tue
"""

import network
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784,30,10])

net.SGD(training_data,30,10,3.0,test_data=test_data)

