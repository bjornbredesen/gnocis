#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################
# Interfacing with TensorFlow

from .features import featureScaler
from .sequences import sequences, positive, negative
from .models import sequenceModel
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random

ntmat = {
	'A': [ 1., 0., 0., 0. ],
	'T': [ 0., 1., 0., 0. ],
	'G': [ 0., 0., 1., 0. ],
	'C': [ 0., 0., 0., 1. ],
	'N': [ 0., 0., 0., 0. ],
}

def getSequenceMatrix(seq):
	return np.array([ ntmat[nt] for nt in seq ])

# Convolutional Neural Network
class sequenceModelCNN(sequenceModel):
	def __init__(self, name, trainingSet, windowSize, windowStep, nConv = 20, convLen = 10, epochs = 100, labelPositive = positive, labelNegative = negative):
		super().__init__(name, enableMultiprocessing = False)
		self.windowSize, self.windowStep = windowSize, windowStep
		self.labelPositive, self.labelNegative = labelPositive, labelNegative
		self.trainingSet = trainingSet
		self.threshold = 0.0
		self.nConv, self.convLen = nConv, convLen
		self.epochs = epochs
		
		positives, negatives = trainingSet.withLabel([ labelPositive, labelNegative ])
		scaleFac = 10
		bloat = int(500/scaleFac)
		hbloat = int(bloat/2)
		
		model = keras.Sequential([
			# Convolutions / Position Weight Matrices (PWMs)
			keras.layers.Conv2D(nConv, (convLen, 4), activation='relu'),
			keras.layers.ZeroPadding2D(padding=(int((convLen-1)/2), 0)),
			keras.layers.Permute((1, 3, 2)),
			# Scale down, for efficiency
			keras.layers.AveragePooling2D((scaleFac, 1)),
			
			#-----------------------------------
			# Motif occurrence combinatorics convolution
			# Extend motif peaks (to model distal combinatorics)
			keras.layers.Conv2D(1, (bloat, 1),
				activation=None,
				trainable = False,
				use_bias=False,
				weights = [
					tf.constant(
						np.array([ 1./bloat for _ in range(bloat) ])
						.reshape((bloat, 1, 1, 1))),
				]),
			keras.layers.ZeroPadding2D(padding=(hbloat, 0)),
			
			#-----------------------------------
			# Convolution for combinatorial motif occurrence modelling
			keras.layers.Conv2D(nConv, (1, nConv), activation='relu'),
			keras.layers.Permute((1, 3, 2)),
			
			#-----------------------------------
			keras.layers.MaxPooling2D((250, 1)),
			keras.layers.Flatten(),
			keras.layers.Dense(2, activation=tf.nn.softmax)
		])
		
		model.compile(
			optimizer = 'adam',
			loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			metrics = ['accuracy']
		)
		
		train_seq = np.array([
			getSequenceMatrix(w.seq).reshape(len(w), 4, 1)
			for s in positives
			for w in s.windows(self.windowSize, self.windowStep)
		] + [
			getSequenceMatrix(w.seq).reshape(len(w), 4, 1)
			for s in negatives
			for w in s.windows(self.windowSize, self.windowStep)
		])

		train_labels = np.array([
			1 for s in positives
		] + [
			0 for s in negatives
		])

		model.fit(
			train_seq,
			train_labels,
			epochs = epochs,
			verbose = 0
		)
		
		self.cls = model

	def scoreWindow(self, seq):
		p = self.cls.predict(np.array([getSequenceMatrix(seq.seq).reshape(len(seq), 4, 1)]))
		return p[0, 1] - p[0, 0]
	
	def getTrainer(self):
		return lambda ts: sequenceModelCNN(self.name, ts, windowSize = self.windowSize, windowStep = self.windowStep, nConv = self.nConv, convLen = self.convLen, epochs = self.epochs, labelPositive = self.labelPositive, labelNegative = self.labelNegative)

	def __str__(self):
		return 'Convolutional Neural Network<Training set: %s; Positive label: %s; Negative label: %s; Convolutions: %d; Convolution length: %d; Epochs: %d>'%(str(self.trainingSet), str(self.labelPositive), str(self.labelNegative), self.nConv, self.convLen, self.epochs)

	def __repr__(self): return self.__str__()

