#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################
# Interfacing with scikit-learn

from .features import featureScaler
from .sequences import sequences
from .models import sequenceModel
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import random

# Support Vector Machines
class sequenceModelSVM(sequenceModel):
	"""
	The `sequenceModelSVM` class trains a Support Vector Machine (SVM) using scikit-learn.
	
	:param name: Model name.
	:param features: Feature set.
	:param positives: Positive training sequences.
	:param negatives: Negative training sequences.
	:param windowSize: Window size.
	:param windowStep: Window step size.
	:param kDegree: Kernel degree.
	
	:type name: str
	:type features: features
	:type positives: sequences
	:type negatives: sequences
	:type windowSize: int
	:type windowStep: int
	:type kDegree: float
	"""
	
	def __init__(self, name, features, positives, negatives, windowSize, windowStep, kDegree):
		super().__init__(name)
		self.features = features
		self.windowSize, self.windowStep = windowSize, windowStep
		self.positives, self.negatives = positives, negatives
		self.kernel = kDegree
		self.threshold = 0.0
		vP = [ self.getSequenceFeatureVector(w) for s in positives for w in s.getWindows(self.windowSize, self.windowStep) ]
		vN = [ self.getSequenceFeatureVector(w) for s in negatives for w in s.getWindows(self.windowSize, self.windowStep) ]
		cP = [ 1.0 for _ in range(len(vP)) ]
		cN = [ -1.0 for _ in range(len(vN)) ]
		self.cls = svm.SVC(kernel = 'poly', degree = kDegree, gamma = 'auto')
		self.cls.fit( np.array(vP+vN), np.array(cP+cN) )
	
	def getTrainer(self):
		return lambda pos, neg: sequenceModelSVM(self.name, self.features, pos, neg, self.windowSize, self.windowStep, self.kernel)
	
	def getSequenceFeatureVector(self, seq):
		return self.features.getAll(seq)
	
	def scoreWindow(self, seq):
		return self.cls.decision_function(np.array([self.getSequenceFeatureVector(seq)]))[0]
	
	def __str__(self):
		return 'Support Vector Machine<Features: %s (%d); Positives: %s; Negatives: %s; Kernel: %s; Support vectors: %d>'%(str(self.features), len(self.features), str(self.positives), str(self.negatives), [ 'linear', 'quadratic', 'cubic' ][self.kernel-1], len(self.cls.support_vectors_))
	
	def __repr__(self): return self.__str__()

# Random Forest
class sequenceModelRF(sequenceModel):
	"""
	The `sequenceModelRF` class trains a Random Forest (RF) model using scikit-learn.
	
	:param name: Model name.
	:param features: Feature set.
	:param positives: Positive training sequences.
	:param negatives: Negative training sequences.
	:param windowSize: Window size.
	:param windowStep: Window step size.
	:param nTrees: Number of trees.
	:param maxDepth: Maximum tree depth.
	
	:type name: str
	:type features: features
	:type positives: sequences
	:type negatives: sequences
	:type windowSize: int
	:type windowStep: int
	:type nTrees: int
	:type maxDepth: int
	"""
	
	def __init__(self, name, features, positives, negatives, windowSize, windowStep, nTrees, maxDepth, scale = True):
		super().__init__(name)
		wPos = sequences(positives.name, [ w for s in positives for w in s.getWindows(self.windowSize, self.windowStep) ])
		wNeg = sequences(positives.name, [ w for s in positives for w in s.getWindows(self.windowSize, self.windowStep) ])
		if scale:
			features = featureScaler( features, positives = wPos, negatives = wNeg )
		self.features = features
		self.windowSize, self.windowStep = windowSize, windowStep
		self.positives, self.negatives = positives, negatives
		self.nTrees, self.maxDepth = nTrees, maxDepth
		self.threshold = 0.0
		vP = [ self.getSequenceFeatureVector(w) for w in wPos ]
		vN = [ self.getSequenceFeatureVector(w) for w in wNeg ]
		cP = [ 1.0 for _ in range(len(vP)) ]
		cN = [ -1.0 for _ in range(len(vN)) ]
		self.cls = RandomForestClassifier(max_depth = maxDepth, random_state = 0)
		self.cls.fit( np.array(vP+vN), np.array(cP+cN) )
	
	def getTrainer(self):
		return lambda pos, neg: sequenceModelRF(self.name, self.features, self.positives, self.negatives, self.windowSize, self.windowStep, self.nTrees, self.maxDepth)
	
	def getSequenceFeatureVector(self, seq):
		return self.features.getAll(seq)
	
	def scoreWindow(self, seq):
		return float(self.cls.predict_proba(np.array([self.getSequenceFeatureVector(seq)]))[0][1])
	
	def __str__(self):
		return 'Random Forest<Features: %s; Positives: %s; Negatives: %s; Trees: %d; Max. depth: %d>'%(str(self.features), str(self.positives), str(self.negatives), self.nTrees, self.maxDepth)
	
	def __repr__(self): return self.__str__()


