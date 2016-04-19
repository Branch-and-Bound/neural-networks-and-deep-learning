# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 14:27:33 2016

@author: Tue Rauff Lind Christensen - Branch & Bound
"""

import network2
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784,30,10])

net.SGD(training_data,30,10,3.0)
