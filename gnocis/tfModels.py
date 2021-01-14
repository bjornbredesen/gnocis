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
	return np.array([ [ ntmat[nt] ] for nt in seq ])

class KerasModel(sequenceModel):
	def __init__(self, name, windowSize, windowStep, modelConstructor, epochs = 100, targetLabel = positive, labels = [ positive, negative ], trainingSet = None, batchsize = 1000, trainWindows = False):
		super().__init__(name, enableMultiprocessing = False)
		self.modelConstructor = modelConstructor
		self.windowSize, self.windowStep = windowSize, windowStep
		self.targetLabel, self.labels = targetLabel, list(set([targetLabel]) | set(labels))
		self.nLabels = len(self.labels)
		self.threshold = 0.0
		self.epochs = epochs
		self.cls = None
		self.batchsize = batchsize
		self.trainingSet = trainingSet
		self.trainWindows = trainWindows
		if trainingSet is not None:
			self._train(trainingSet)

	def _train(self, trainingSet):
		model = self.modelConstructor()
		
		#----------------------------------------------------------

		model.compile(
			optimizer = 'adam',
			loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			metrics = ['accuracy']
		)

		self.labelValues = {
			label: i
			for i, label in enumerate(self.labels)
		}
		if self.trainWindows:
			train_seq = np.array([
				getSequenceMatrix(win.seq)
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
		else:
			train_seq = np.array([
				getSequenceMatrix(seq.seq)
				for label in self.labels
				for seq in trainingSet.withLabel(label)
			])
			train_labels = np.array([
				self.labelValues[label]
				for label in self.labels
				for seq in trainingSet.withLabel(label)
			])

		model.fit(
			train_seq,
			train_labels,
			epochs = self.epochs,
			verbose = 0
		)

		self.cls = model

	def getSequenceScores(self, seqs, nStreamFetch = 100000, maxStreamFetchNT = 100000000):
		if self.batchsize == 0:
			return super().getSequenceScores(seqs)
		if isinstance(seqs, sequenceStream):
			def fetchseq():
				i = 0
				for blk in seqs.fetch(nStreamFetch, maxStreamFetchNT):
					for cseq in blk:
						for win in cseq.windows(self.windowSize, self.windowStep):
							if len(win) != self.windowSize:
								continue
							#
							yield (i, win)
						i += 1
			#
			seqwinit = (
				(i, win)
				for i, win in fetchseq()
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
			p = self.cls.predict(np.array([getSequenceMatrix(seq.seq) for i, seq in batch]))
			i = self.labelValues[self.targetLabel]
			scores = p[:, i]
			#scores = p[:, i] * p[:, i] / (sum(p[:, n] for n in range(self.nLabels)) + 0.1) # Pseudocount to avoid zero division
			for score, (i, win) in zip(scores, batch):
				if i >= len(seqscores):
					seqscores += [ -float('INF') for _ in range(i - len(seqscores) + 1) ]
				if score > seqscores[i]:
					seqscores[i] = score
		#
		return seqscores

	def scoreWindow(self, seq):
		p = self.cls.predict(np.array([ getSequenceMatrix(seq.seq) ]))
		i = self.labelValues[self.targetLabel]
		return p[0, i]
		#return p[0, i] * p[0, i] / (sum(p[0, n] for n in range(self.nLabels)) + 0.1)

	def getTrainer(self):
		return lambda ts: KerasModel(self.name, trainingSet = ts, windowSize = self.windowSize, windowStep = self.windowStep, epochs = self.epochs, targetLabel = self.targetLabel, labels = self.labels, batchsize = self.batchsize, modelConstructor = self.modelConstructor)

	def __str__(self):
		return 'Multi-class Keras model <Training set: %s; Labels: %s; Target label: %s; Epochs: %d>'%(str(self.trainingSet), str(self.labels), str(self.targetLabel), self.epochs)

	def __repr__(self): return self.__str__()

class DeepMOCCA(KerasModel):
	def __init__(self, name, windowSize, windowStep, nConv = 20, convLen = 10, epochs = 100,
				 targetLabel = positive, labels = [ positive, negative ], trainingSet = None,
				 batchsize = 1000, trainWindows = True,
				 includeDinucleotides = True,
				 pairingDistance = 500):
		self.nConv, self.convLen = nConv, convLen
		self.includeDinucleotides = includeDinucleotides
		self.pairingDistance = pairingDistance
		modelConstructor = lambda: self.constructModel(windowSize = windowSize,
						nConv = nConv, convLen = convLen, labels = labels)
		super().__init__(name = name, windowSize = windowSize, windowStep = windowStep,
						modelConstructor = modelConstructor, epochs = epochs,
						targetLabel = targetLabel, labels = labels,
						trainingSet = trainingSet,
						batchsize = batchsize, trainWindows = trainWindows)

	def constructModel(self, windowSize, nConv, convLen, labels):
		scaleFac = 10
		bloat = int((min(self.pairingDistance, windowSize)-1)/scaleFac)
		hbloat = int(bloat/2)
		nLabels = len(labels)

		#----------------------------------------------------------
		# Model structure
		I = keras.Input(shape = (None, 1, 4))

		# - Merges down NT channels and outputs nConv convolutions
		C1 = keras.layers.Conv2D(nConv, kernel_size = (convLen, 1), activation = 'relu',
									  data_format = "channels_last")(I)
		
		if self.includeDinucleotides:
			C1P = keras.layers.ZeroPadding2D(padding = (int(convLen / 2) - 1, 0))(C1)

			# Dinucleotide convolutions
			# - Merges down NT channels and outputs nConv convolutions
			C2 = keras.layers.Conv2D(nConv, kernel_size = (2, 1), activation = 'relu',
										  data_format = "channels_last")(I)
			C2P = C2

			Combine = keras.layers.Concatenate(axis=-1)([ C1P, C2P ])
		else:
			Combine = C1

		# Reduces resolution (but not channels/convolutions)
		DS = keras.layers.AveragePooling2D((scaleFac, 1))(Combine)

		# Bloating - Averages convolution outputs over a window
		BLT0 = keras.layers.Permute((1, 3, 2))(DS)
		BLT1 = keras.layers.Conv2D(1, kernel_size = (bloat, 1),
									  activation=None,
									  trainable = False,
									  use_bias=False,
									  weights = [
										tf.constant(
										   np.array([ 1./bloat for _ in range(bloat) ])
											 .reshape((bloat, 1, 1, 1))),
									  ],
									  data_format = "channels_last")(BLT0)
		BLT = keras.layers.Permute((1, 3, 2))(BLT1)

		# Merges convolutions into mixing convolutions
		C2 = keras.layers.Conv2D(nConv, kernel_size = (1, 1), activation = 'relu',
									  data_format = "channels_last")(BLT)
		Pool = keras.layers.GlobalMaxPooling2D(data_format = "channels_last")(C2)

		# Max pooling of mixing convolutions
		Term = keras.layers.Dense(nLabels, activation=tf.nn.softmax)(Pool)

		model = keras.Model(I, Term)
		
		return model
		
	def getTrainer(self):
		return lambda ts: DeepMOCCA(self.name,
			trainingSet = ts, windowSize = self.windowSize,
			windowStep = self.windowStep, nConv = self.nConv,
			convLen = self.convLen, epochs = self.epochs,
			targetLabel = self.targetLabel, labels = self.labels,
			batchsize = self.batchsize, trainWindows = self.trainWindows,
			includeDinucleotides = self.includeDinucleotides,
			pairingDistance = self.pairingDistance)

	def __str__(self):
		return 'Deep-MOCCA <Training set: %s; Labels: %s; Target label: %s; Convolutions: %d; Convolution length: %d; Pairing distance: %d; Include dinucleotides: %s; Epochs: %d>'%(str(self.trainingSet), str(self.labels), str(self.targetLabel), self.nConv, self.convLen, self.pairingDistance, 'Yes' if self.pairingDistance else 'No', self.epochs)

	def __repr__(self): return self.__str__()

