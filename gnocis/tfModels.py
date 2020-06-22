#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################
# Interfacing with TensorFlow

from .sequences import sequences, positive, negative, sequenceStream
from .models import sequenceModel
import numpy as np
import tensorflow as tf
from tensorflow import keras

def setSeed(seed):
	tf.random.set_seed(seed)

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
	def __init__(self, name, windowSize, windowStep, nConv = 20, convLen = 10, epochs = 100, labelPositive = positive, labelNegative = negative, trainingSet = None, batchsize = 1000):
		super().__init__(name, enableMultiprocessing = False)
		self.windowSize, self.windowStep = windowSize, windowStep
		self.labelPositive, self.labelNegative = labelPositive, labelNegative
		self.threshold = 0.0
		self.nConv, self.convLen = nConv, convLen
		self.epochs = epochs
		self.cls = None
		self.batchsize = batchsize
		if trainingSet is not None:
			self.train(trainingSet)
	
	def train(self, trainingSet):
		self.trainingSet = trainingSet
		positives, negatives = trainingSet.withLabel([ self.labelPositive, self.labelNegative ])
		scaleFac = 10
		bloat = int((min(500, self.windowSize)-1)/scaleFac)
		hbloat = int(bloat/2)
		
		model = keras.Sequential([
			# Convolutions / Position Weight Matrices (PWMs)
			keras.layers.Conv2D(self.nConv, (self.convLen, 4), activation='relu'),
			keras.layers.ZeroPadding2D(padding=(int((self.convLen-1)/2), 0)),
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
			keras.layers.Conv2D(self.nConv, (1, self.nConv), activation='relu'),
			keras.layers.Permute((1, 3, 2)),
			
			#-----------------------------------
			keras.layers.MaxPooling2D((int(250/scaleFac), 1)),
			keras.layers.Flatten(),
			keras.layers.Dense(2, activation=tf.nn.softmax)
		])
		
		model.compile(
			optimizer = 'adam',
			loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			metrics = ['accuracy']
		)
		
		train_seq = [
			getSequenceMatrix(w.seq).reshape(len(w), 4, 1)
			for s in negatives
			for w in s.windows(self.windowSize, self.windowStep)
		] + np.array([
			getSequenceMatrix(w.seq).reshape(len(w), 4, 1)
			for s in positives
			for w in s.windows(self.windowSize, self.windowStep)
		])

		train_labels = [
			0 for s in negatives
			for w in s.windows(self.windowSize, self.windowStep)
		] + np.array([
			1 for s in positives
			for w in s.windows(self.windowSize, self.windowStep)
		])

		model.fit(
			train_seq,
			train_labels,
			epochs = self.epochs,
			verbose = 0
		)
		
		self.cls = model
	
	def getSequenceScores(self, seqs):
		if self.batchsize == 0:
			return super().getSequenceScores(seqs)
		if isinstance(seqs, sequenceStream):
			nTreadFetch = 100000
			maxThreadFetchNT = nTreadFetch * 1000
			seqwinit = (
				(i, win)
				for blk in seqs.fetch(nTreadFetch, maxThreadFetchNT)
				for i, cseq in enumerate(blk)
				for win in cseq.windows(self.windowSize, self.windowStep)
				if len(win) == self.windowSize
			)
		else:
			seqwinit = (
				(i, win)
				for i, cseq in enumerate(seqs)
				for win in cseq.windows(self.windowSize, self.windowStep)
				if len(win) == self.windowSize
			)
		seqscores = [  ]
		while True:
			batch = [
				b for b in [
					next(seqwinit, None) for _ in range(self.batchsize)
				]
				if b is not None
			]
			if len(batch) == 0: break
			p = self.cls.predict(np.array([getSequenceMatrix(seq.seq).reshape(len(seq), 4, 1) for i, seq in batch]))
			scores = p[:, 1] - p[:, 0]
			for score, (i, win) in zip(scores, batch):
				if i >= len(seqscores):
					seqscores += [ -float('INF') for _ in range(i - len(seqscores) + 1) ]
				if score > seqscores[i]:
					seqscores[i] = score
		#
		return seqscores
	
	def scoreWindow(self, seq):
		p = self.cls.predict(np.array([getSequenceMatrix(seq.seq).reshape(len(seq), 4, 1)]))
		return p[0, 1] - p[0, 0]
	
	def getTrainer(self):
		return lambda ts: sequenceModelCNN(self.name, trainingSet = ts, windowSize = self.windowSize, windowStep = self.windowStep, nConv = self.nConv, convLen = self.convLen, epochs = self.epochs, labelPositive = self.labelPositive, labelNegative = self.labelNegative, batchsize = self.batchsize)

	def __str__(self):
		return 'Convolutional Neural Network<Training set: %s; Positive label: %s; Negative label: %s; Convolutions: %d; Convolution length: %d; Epochs: %d>'%(str(self.trainingSet), str(self.labelPositive), str(self.labelNegative), self.nConv, self.convLen, self.epochs)

	def __repr__(self): return self.__str__()

# Convolutional Neural Network
class sequenceModelMultiCNN(sequenceModel):
	def __init__(self, name, windowSize, windowStep, nConv = 20, convLen = 10, epochs = 100, targetLabel = positive, labels = [ positive, negative ], trainingSet = None, batchsize = 1000):
		super().__init__(name, enableMultiprocessing = False)
		self.windowSize, self.windowStep = windowSize, windowStep
		self.targetLabel, self.labels = targetLabel, list(set([targetLabel]) | set(labels))
		self.nLabels = len(self.labels)
		self.threshold = 0.0
		self.nConv, self.convLen = nConv, convLen
		self.epochs = epochs
		self.cls = None
		self.batchsize = batchsize
		if trainingSet is not None:
			self.train(trainingSet)
	
	def train(self, trainingSet):
		self.trainingSet = trainingSet
		scaleFac = 10
		bloat = int((min(500, self.windowSize)-1)/scaleFac)
		hbloat = int(bloat/2)
		
		model = keras.Sequential([
			# Convolutions / Position Weight Matrices (PWMs)
			keras.layers.Conv2D(self.nConv, (self.convLen, 4), activation='relu'),
			keras.layers.ZeroPadding2D(padding=(int((self.convLen-1)/2), 0)),
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
			keras.layers.Conv2D(self.nConv, (1, self.nConv), activation='relu'),
			keras.layers.Permute((1, 3, 2)),
			
			#-----------------------------------
			keras.layers.MaxPooling2D((int(250/scaleFac), 1)),
			keras.layers.Flatten(),
			keras.layers.Dense(self.nLabels, activation=tf.nn.softmax)
		])
		
		model.compile(
			optimizer = 'adam',
			loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			metrics = ['accuracy']
		)
		
		self.labelValues = {
			label: i
			for i, label in enumerate(self.labels)
		}
		train_seq = np.array([
			getSequenceMatrix(win.seq).reshape(len(win), 4, 1)
			for label in self.labels
			for seq in trainingSet.withLabel(label)
			for win in seq.windows(self.windowSize, self.windowStep)
		])
		train_labels = np.array([
			self.labelValues[label]
			for label in self.labels
			for seq in trainingSet.withLabel(label)
			for win in seq.windows(self.windowSize, self.windowStep)
		])

		model.fit(
			train_seq,
			train_labels,
			epochs = self.epochs,
			verbose = 0
		)
		
		self.cls = model
	
	def getSequenceScores(self, seqs):
		if self.batchsize == 0:
			return super().getSequenceScores(seqs)
		if isinstance(seqs, sequenceStream):
			nTreadFetch = 100000
			maxThreadFetchNT = nTreadFetch * 1000
			seqwinit = (
				(i, win)
				for blk in seqs.fetch(nTreadFetch, maxThreadFetchNT)
				for i, cseq in enumerate(blk)
				for win in cseq.windows(self.windowSize, self.windowStep)
				if len(win) == self.windowSize
			)
		else:
			seqwinit = (
				(i, win)
				for i, cseq in enumerate(seqs)
				for win in cseq.windows(self.windowSize, self.windowStep)
				if len(win) == self.windowSize
			)
		seqscores = [  ]
		while True:
			batch = [
				b for b in [
					next(seqwinit, None) for _ in range(self.batchsize)
				]
				if b is not None
			]
			if len(batch) == 0: break
			p = self.cls.predict(np.array([getSequenceMatrix(seq.seq).reshape(len(seq), 4, 1) for i, seq in batch]))
			#scores = p[:, 1] - p[:, 0]
			i = self.labelValues[self.targetLabel]
			scores = p[:, i] / sum(p[:, n] for n in range(self.nLabels))
			for score, (i, win) in zip(scores, batch):
				if i >= len(seqscores):
					seqscores += [ -float('INF') for _ in range(i - len(seqscores) + 1) ]
				if score > seqscores[i]:
					seqscores[i] = score
		#
		return seqscores
	
	def scoreWindow(self, seq):
		p = self.cls.predict(np.array([getSequenceMatrix(seq.seq).reshape(len(seq), 4, 1)]))
		i = self.labelValues[self.targetLabel]
		return p[0, i] / sum(p[0, n] for n in range(self.nLabels))
	
	def getTrainer(self):
		return lambda ts: sequenceModelMultiCNN(self.name, trainingSet = ts, windowSize = self.windowSize, windowStep = self.windowStep, nConv = self.nConv, convLen = self.convLen, epochs = self.epochs, targetLabel = self.targetLabel, labels = self.labels, batchsize = self.batchsize)

	def __str__(self):
		return 'Multi-class Convolutional Neural Network<Training set: %s; Positive label: %s; Negative label: %s; Convolutions: %d; Convolution length: %d; Epochs: %d>'%(str(self.trainingSet), str(self.labelPositive), str(self.labelNegative), self.nConv, self.convLen, self.epochs)

	def __repr__(self): return self.__str__()


# Convolutional Neural Network
class sequenceModelKeras(sequenceModel):
	def __init__(self, name, windowSize, windowStep, kerasModel, epochs = 100, targetLabel = positive, labels = [ positive, negative ], trainingSet = None, batchsize = 1000):
		super().__init__(name, enableMultiprocessing = False)
		self.windowSize, self.windowStep = windowSize, windowStep
		self.targetLabel, self.labels = targetLabel, list(set([targetLabel]) | set(labels))
		self.nLabels = len(self.labels)
		self.threshold = 0.0
		self.kerasModel = kerasModel
		self.epochs = epochs
		self.cls = None
		self.batchsize = batchsize
		if trainingSet is not None:
			self.train(trainingSet)
	
	def train(self, trainingSet):
		self.trainingSet = trainingSet
		scaleFac = 10
		bloat = int((min(500, self.windowSize)-1)/scaleFac)
		hbloat = int(bloat/2)
		
		model = self.kerasModel
		
		model.compile(
			optimizer = 'adam',
			loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			metrics = ['accuracy']
		)
		
		self.labelValues = {
			label: i
			for i, label in enumerate(self.labels)
		}
		train_seq = np.array([
			getSequenceMatrix(win.seq).reshape(len(win), 4, 1)
			for label in self.labels
			for seq in trainingSet.withLabel(label)
			for win in seq.windows(self.windowSize, self.windowStep)
		])
		train_labels = np.array([
			self.labelValues[label]
			for label in self.labels
			for seq in trainingSet.withLabel(label)
			for win in seq.windows(self.windowSize, self.windowStep)
		])

		model.fit(
			train_seq,
			train_labels,
			epochs = self.epochs,
			verbose = 0
		)
		
		self.cls = model
	
	def getSequenceScores(self, seqs):
		if self.batchsize == 0:
			return super().getSequenceScores(seqs)
		if isinstance(seqs, sequenceStream):
			nTreadFetch = 100000
			maxThreadFetchNT = nTreadFetch * 1000
			seqwinit = (
				(i, win)
				for blk in seqs.fetch(nTreadFetch, maxThreadFetchNT)
				for i, cseq in enumerate(blk)
				for win in cseq.windows(self.windowSize, self.windowStep)
				if len(win) == self.windowSize
			)
		else:
			seqwinit = (
				(i, win)
				for i, cseq in enumerate(seqs)
				for win in cseq.windows(self.windowSize, self.windowStep)
				if len(win) == self.windowSize
			)
		seqscores = [  ]
		while True:
			batch = [
				b for b in [
					next(seqwinit, None) for _ in range(self.batchsize)
				]
				if b is not None
			]
			if len(batch) == 0: break
			p = self.cls.predict(np.array([getSequenceMatrix(seq.seq).reshape(len(seq), 4, 1) for i, seq in batch]))
			#scores = p[:, 1] - p[:, 0]
			i = self.labelValues[self.targetLabel]
			scores = p[:, i] / sum(p[:, n] for n in range(self.nLabels))
			for score, (i, win) in zip(scores, batch):
				if i >= len(seqscores):
					seqscores += [ -float('INF') for _ in range(i - len(seqscores) + 1) ]
				if score > seqscores[i]:
					seqscores[i] = score
		#
		return seqscores
	
	def scoreWindow(self, seq):
		p = self.cls.predict(np.array([getSequenceMatrix(seq.seq).reshape(len(seq), 4, 1)]))
		i = self.labelValues[self.targetLabel]
		return p[0, i] / sum(p[0, n] for n in range(self.nLabels))
	
	def getTrainer(self):
		return lambda ts: sequenceModelKeras(self.name, trainingSet = ts, windowSize = self.windowSize, windowStep = self.windowStep, kerasModel = self.kerasModel, epochs = self.epochs, targetLabel = self.targetLabel, labels = self.labels, batchsize = self.batchsize)

	def __str__(self):
		return 'Multi-class Keras Neural Network<Training set: %s; Positive label: %s; Negative label: %s; Epochs: %d>'%(str(self.trainingSet), str(self.labelPositive), str(self.labelNegative), self.epochs)

	def __repr__(self): return self.__str__()

