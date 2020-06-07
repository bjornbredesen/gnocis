#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################
# Interfacing with scikit-learn

from .features import featureScaler
from .sequences import sequences, positive, negative
from .models import sequenceModel
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.linear_model import Lasso
from .featurenetwork import baseModel

# Support Vector Machines - Sequence model
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
	
	def __init__(self, name, features, trainingSet, windowSize, windowStep, kDegree, scale = True, labelPositive = positive, labelNegative = negative):
		super().__init__(name)
		self.windowSize, self.windowStep = windowSize, windowStep
		self.labelPositive, self.labelNegative = labelPositive, labelNegative
		self.trainingSet = trainingSet
		positives, negatives = trainingSet.withLabel([ labelPositive, labelNegative ])
		wPos = sequences(positives.name, [ w for s in positives for w in s.windows(self.windowSize, self.windowStep) ])
		wNeg = sequences(negatives.name, [ w for s in negatives for w in s.windows(self.windowSize, self.windowStep) ])
		if scale:
			features = featureScaler( features, trainingSet = wPos + wNeg )
		self.scale = scale
		self.features = features
		self.kernel = kDegree
		self.threshold = 0.0
		vP = [ self.getSequenceFeatureVector(w) for w in wPos ]
		vN = [ self.getSequenceFeatureVector(w) for w in wNeg ]
		cP = [ 1.0 for _ in range(len(vP)) ]
		cN = [ -1.0 for _ in range(len(vN)) ]
		self.cls = svm.SVC(kernel = 'poly', degree = kDegree, gamma = 'auto')
		self.cls.fit( np.array(vP+vN), np.array(cP+cN) )
	
	def getTrainer(self):
		return lambda ts: sequenceModelSVM(self.name, self.features, ts, windowSize = self.windowSize, windowStep = self.windowStep, kDegree = self.kernel, scale = self.scale, labelPositive = self.labelPositive, labelNegative = self.labelNegative)
	
	def getSequenceFeatureVector(self, seq):
		return self.features.getAll(seq)
	
	def scoreWindow(self, seq):
		return self.cls.decision_function(np.array([self.getSequenceFeatureVector(seq)]))[0]
	
	def __str__(self):
		return 'Support Vector Machine<Features: %s (%d); Training set: %s; Positive label: %s; Negative label: %s; Kernel: %s; Support vectors: %d>'%(str(self.features), len(self.features), str(self.trainingSet), str(self.labelPositive), str(self.labelNegative), [ 'linear', 'quadratic', 'cubic' ][self.kernel-1], len(self.cls.support_vectors_))
	
	def __repr__(self): return self.__str__()

# Random Forest - Sequence model
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
	
	def __init__(self, name, features, trainingSet, windowSize, windowStep, nTrees = 100, maxDepth = None, scale = True, labelPositive = positive, labelNegative = negative):
		super().__init__(name)
		self.windowSize, self.windowStep = windowSize, windowStep
		self.labelPositive, self.labelNegative = labelPositive, labelNegative
		self.trainingSet = trainingSet
		positives, negatives = trainingSet.withLabel([ labelPositive, labelNegative ])
		wPos = sequences(positives.name, [ w for s in positives for w in s.windows(self.windowSize, self.windowStep) ])
		wNeg = sequences(negatives.name, [ w for s in negatives for w in s.windows(self.windowSize, self.windowStep) ])
		if scale:
			features = featureScaler( features, trainingSet = wPos + wNeg )
		self.scale = scale
		self.features = features
		self.nTrees, self.maxDepth = nTrees, maxDepth
		self.threshold = 0.0
		vP = [ self.getSequenceFeatureVector(w) for w in wPos ]
		vN = [ self.getSequenceFeatureVector(w) for w in wNeg ]
		cP = [ 1.0 for _ in range(len(vP)) ]
		cN = [ -1.0 for _ in range(len(vN)) ]
		self.cls = RandomForestClassifier(n_estimators = nTrees, max_depth = maxDepth, random_state = 0)
		self.cls.fit( np.array(vP+vN), np.array(cP+cN) )
	
	def getTrainer(self):
		return lambda ts: sequenceModelRF(self.name, self.features, ts, windowSize = self.windowSize, windowStep = self.windowStep, nTrees = self.nTrees, maxDepth = self.maxDepth, scale = self.scale, labelPositive = self.labelPositive, labelNegative = self.labelNegative)
	
	def getSequenceFeatureVector(self, seq):
		return self.features.getAll(seq)
	
	def scoreWindow(self, seq):
		return float(self.cls.predict_proba(np.array([self.getSequenceFeatureVector(seq)]))[0][1])
	
	def __str__(self):
		return 'Random Forest<Features: %s; Training set: %s; Positive label: %s; Negative label: %s; Trees: %d; Max. depth: %s>'%(str(self.features), str(self.trainingSet), str(self.labelPositive), str(self.labelNegatives), self.nTrees, str(self.maxDepth))
	
	def __repr__(self): return self.__str__()

# Lasso
class sequenceModelLasso(sequenceModel):
	"""
	The `sequenceModelLasso` class trains a Lasso model using scikit-learn.
	
	:param name: Model name.
	:param features: Feature set.
	:param positives: Positive training sequences.
	:param negatives: Negative training sequences.
	:param windowSize: Window size.
	:param windowStep: Window step size.
	:param alpha: Alpha parameter for Lasso.
	
	:type name: str
	:type features: features
	:type positives: sequences
	:type negatives: sequences
	:type windowSize: int
	:type windowStep: int
	:type alpha: float
	"""
	
	def __init__(self, name, features, trainingSet, windowSize, windowStep, alpha = 1., labelPositive = positive, labelNegative = negative):
		super().__init__(name)
		self.windowSize, self.windowStep = windowSize, windowStep
		self.labelPositive, self.labelNegative = labelPositive, labelNegative
		self.trainingSet = trainingSet
		positives, negatives = trainingSet.withLabel([ labelPositive, labelNegative ])
		wPos = sequences(positives.name, [ w for s in positives for w in s.windows(self.windowSize, self.windowStep) ])
		wNeg = sequences(negatives.name, [ w for s in negatives for w in s.windows(self.windowSize, self.windowStep) ])
		self.features = features
		self.threshold = 0.0
		vP = [ self.getSequenceFeatureVector(w) for w in wPos ]
		vN = [ self.getSequenceFeatureVector(w) for w in wNeg ]
		cP = [ 1.0 for _ in range(len(vP)) ]
		cN = [ -1.0 for _ in range(len(vN)) ]
		self.alpha = alpha
		self.cls = Lasso(alpha = alpha)
		self.cls.fit( np.array(vP+vN), np.array(cP+cN) )
	
	def getTrainer(self):
		return lambda ts: sequenceModelLasso(self.name, self.features, ts, windowSize = self.windowSize, windowStep = self.windowStep, alpha = self.alpha, labelPositive = self.labelPositive, labelNegative = self.labelNegative)
	
	def getSequenceFeatureVector(self, seq):
		return self.features.getAll(seq)
	
	def scoreWindow(self, seq):
		return self.cls.predict(np.array([self.getSequenceFeatureVector(seq)]))[0]
	
	def __str__(self):
		return 'Lasso<Features: %s; Training set: %s; Positive label: %s; Negative label: %s; Alpha: %f>'%(str(self.features), str(self.trainingSet), str(self.labelPositive), str(self.labelNegative), self.alpha)
	
	def __repr__(self): return self.__str__()

# Support Vector Machines - Base model
class SVM(baseModel):
	
	def __init__(self, model = None, labelPositive = positive, labelNegative = negative, kDegree = 1, C = 4):
		super().__init__()
		self.labelPositive, self.labelNegative = labelPositive, labelNegative
		self.labelNegative = labelNegative
		self.C, self.kDegree = C, kDegree
		self.mdl = model
	
	def __str__(self):
		return 'Support Vector Machine<C: %s; Kernel degree: %s; Positive label: %s; Negative label: %s>'%(str(self.C), str(self.kDegree), str(self.labelPositive), str(self.labelNegative))
	
	def score(self, featureVectors):
		return [
			[ self.mdl.decision_function(np.array([fv]))[0] ]
			for fv in featureVectors
		]
	
	def train(self, trainingSet):
		fvPos = trainingSet[self.labelPositive]
		fvNeg = trainingSet[self.labelNegative]
		cP = [ 1.0 for _ in range(len(fvPos)) ]
		cN = [ -1.0 for _ in range(len(fvNeg)) ]
		mdl = svm.SVC(C = self.C, kernel = 'poly', degree = self.kDegree, gamma = 'auto')
		mdl.fit( np.array(fvPos+fvNeg), np.array(cP+cN) )
		return SVM(
			model = mdl,
			labelPositive = self.labelPositive,
			labelNegative = self.labelNegative,
			C = self.C,
			kDegree = self.kDegree)
	
	def weights(self, featureNames):
		return nctable(
			'Weights: ' + str(self),
			{
				**{
					'Feature': self.featureNames
				},
				**{
					'Weight': self._weights
				}
			},
			align = { 'Feature': 'l' }
		)
	
	def __repr__(self): return self.__str__()

# Random Forest - Base model
class RF(baseModel):
	
	def __init__(self, model = None, labelPositive = positive, labelNegative = negative, nTrees = 100, maxDepth = None):
		super().__init__()
		self.labelPositive, self.labelNegative = labelPositive, labelNegative
		self.labelNegative = labelNegative
		self.nTrees, self.maxDepth = nTrees, maxDepth
		self.mdl = model
	
	def __str__(self):
		return 'Random Forest<Trees: %s; Max. depth: %s; Positive label: %s; Negative label: %s>'%(str(self.nTrees), str(self.maxDepth), str(self.labelPositive), str(self.labelNegative))
	
	def score(self, featureVectors):
		return [
			#[ self.mdl.decision_function(np.array([fv]))[0] ]
			[ float(self.mdl.predict_proba(np.array([fv]))[0][1]) ]
			for fv in featureVectors
		]
	
	def train(self, trainingSet):
		fvPos = trainingSet[self.labelPositive]
		fvNeg = trainingSet[self.labelNegative]
		cP = [ 1.0 for _ in range(len(fvPos)) ]
		cN = [ -1.0 for _ in range(len(fvNeg)) ]
		#mdl = svm.SVC(C = self.C, kernel = 'poly', degree = self.kDegree, gamma = 'auto')
		mdl = RandomForestClassifier(n_estimators = self.nTrees, max_depth = self.maxDepth, random_state = 0)
		mdl.fit( np.array(fvPos+fvNeg), np.array(cP+cN) )
		return RF(
			model = mdl,
			labelPositive = self.labelPositive,
			labelNegative = self.labelNegative,
			nTrees = self.nTrees,
			maxDepth = self.maxDepth)
	
	def weights(self, featureNames):
		return None
	
	def __repr__(self): return self.__str__()


