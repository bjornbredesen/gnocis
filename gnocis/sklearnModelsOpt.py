#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################
# Interfacing with scikit-learn

from .features import featureScaler
from .models import sequenceModel
from .sequences import sequences, positive, negative
import numpy as np
import cupy as cp
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import random

# Support Vector Machines
class sequenceModelSVMOptimizedQuadratic(sequenceModel):
	"""
	The `sequenceModelSVMOptimizedQuadratic` class trains a quadratic kernel Support Vector Machine (SVM) using scikit-learn. The kernel is applied using matrix multiplication.
	
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
		self.windowSize, self.windowStep = windowSize, windowStep
		self.kernel = kDegree
		self.threshold = 0.0
		assert(kDegree == 2)
		vP = [ self.getSequenceFeatureVector(w) for w in wPos ]
		vN = [ self.getSequenceFeatureVector(w) for w in wNeg ]
		cP = [ 1.0 for _ in range(len(vP)) ]
		cN = [ -1.0 for _ in range(len(vN)) ]
		cls = svm.SVC(kernel = 'poly', degree = kDegree, gamma = 'auto')
		cls.fit( np.array(vP+vN), np.array(cP+cN) )
		assert(cls.coef0 == 0.0)
		# Extract basic kernel values
		SV = cls.support_vectors_
		nfeatures = len(cls.support_vectors_[0])
		bias = cls.intercept_[0]
		c0 = cls.coef0
		c = cls.dual_coef_[0]
		gamma = 1.0 / (nfeatures)
		self.nSV = len(cls.support_vectors_)
		# Extract weight matrix
		self.pairWeights = ((SV.T * c) @ SV) * gamma * gamma
		# Extract Bias
		self.bias = -bias
	
	def getTrainer(self):
		return lambda ts: sequenceModelSVMOptimizedQuadratic(self.name, self.features, ts, windowSize = self.windowSize, windowStep = self.windowStep, alpha = self.alpha, labelPositive = self.labelPositive, labelNegative = self.labelNegative)
	
	def getSequenceFeatureVector(self, seq):
		return self.features.getAll(seq)
	
	def scoreWindow(self, seq):
		fv = np.array(self.getSequenceFeatureVector(seq))
		return self.pairWeights @ fv @ fv - self.bias
	
	def __str__(self):
		return 'Support Vector Machine<Features: %s (%d); Training set: %s; Positive label: %s; Negative label; Kernel: %s; Support vectors: %d; Quadratic optimized>'%(str(self.features), len(self.features), str(self.trainingSet), str(self.labelPositive), str(self.labelNegative), [ 'linear', 'quadratic', 'cubic' ][self.kernel-1], self.nSV)
	
	def __repr__(self): return self.__str__()


# Support Vector Machines
class sequenceModelSVMOptimizedQuadraticAutoScale(sequenceModel):
	"""
	The `sequenceModelSVMOptimizedQuadratic` class trains a quadratic kernel Support Vector Machine (SVM) using scikit-learn. The kernel is applied using matrix multiplication.
	
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
		assert(kDegree == 2)
		vP = [ self.getSequenceFeatureVector(w) for w in wPos ]
		vN = [ self.getSequenceFeatureVector(w) for w in wNeg ]
		cP = [ 1.0 for _ in range(len(vP)) ]
		cN = [ -1.0 for _ in range(len(vN)) ]
		cls = svm.SVC(kernel = 'poly', degree = kDegree, gamma = 'auto')
		cls.fit( np.array(vP+vN), np.array(cP+cN) )
		assert(cls.coef0 == 0.0)
		# Extract basic kernel values
		SV = cls.support_vectors_
		nfeatures = len(cls.support_vectors_[0])
		bias = cls.intercept_[0]
		c0 = cls.coef0
		c = cls.dual_coef_[0]
		gamma = 1.0 / (nfeatures)
		self.nSV = len(cls.support_vectors_)
		# Extract weight matrix
		alpha = [ ( f.vSub - 1.0 ) / f.vScale for f in features ]
		beta = [ ( f.vSub + 1.0 ) / f.vScale for f in features ]
		#
		self.smat = np.array([
			[
				(1./(beta[i]-alpha[i])) * (1./(beta[j]-alpha[j]))
				for j in range(len(features))
			]
			for i in range(len(features))
		])
		SVProd = ( (SV.T * c) @ SV ) * gamma * gamma
		self.pairWeights = SVProd * 4. * self.smat
		#
		self.linWeights = np.array([
			sum(
				-4. * SVProd[i, j] * self.smat[i, j] * (alpha[j]+beta[j])
				for j in range(len(features))
			)
			for i in range(len(features))
		])
		#
		self.bias = sum(
						gamma * gamma * _c * (sum(
							_v[i] * (alpha[i]+beta[i]) / (beta[i] - alpha[i])
							for i in range(len(features))
						)**2.)
					for _v, _c in zip(SV, c)
				) + cls.intercept_[0]
	
	def getTrainer(self):
		return lambda ts: sequenceModelSVMOptimizedQuadraticAutoScale(self.name, self.features, ts, windowSize = self.windowSize, windowStep = self.windowStep, alpha = self.alpha, labelPositive = self.labelPositive, labelNegative = self.labelNegative)
	
	def getSequenceFeatureVector(self, seq):
		return self.features.getAll(seq)
	
	def scoreWindow(self, seq):
		fraw = nc.features('', [ f.feature for f in self.features ])
		fv = np.array(fraw.getAll(seq))
		return (self.pairWeights @ fv @ fv) + (self.linWeights @ fv) + self.bias
	
	def __str__(self):
		return 'Support Vector Machine<Features: %s (%d); Training set: %s; Positive label: %s; Negative label; Kernel: %s; Support vectors: %d; Quadratic optimized>'%(str(self.features), len(self.features), str(self.trainingSet), str(self.labelPositive), str(self.labelNegative), [ 'linear', 'quadratic', 'cubic' ][self.kernel-1], self.nSV)
	
	def __repr__(self): return self.__str__()


# Support Vector Machines
class sequenceModelSVMOptimizedQuadraticCUDA(sequenceModel):
	"""
	The `sequenceModelSVMOptimizedQuadraticCUDA` class trains a quadratic kernel Support Vector Machine (SVM) using scikit-learn. The kernel is applied using matrix multiplication with CUDA.
	
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
		assert(kDegree == 2)
		vP = [ self.getSequenceFeatureVector(w) for w in wPos ]
		vN = [ self.getSequenceFeatureVector(w) for w in wNeg ]
		cP = [ 1.0 for _ in range(len(vP)) ]
		cN = [ -1.0 for _ in range(len(vN)) ]
		cls = svm.SVC(kernel = 'poly', degree = kDegree, gamma = 'auto')
		cls.fit( np.array(vP+vN), np.array(cP+cN) )
		assert(cls.coef0 == 0.0)
		# Extract basic kernel values
		SV = cls.support_vectors_
		nfeatures = len(cls.support_vectors_[0])
		bias = cls.intercept_[0]
		c0 = cls.coef0
		c = cls.dual_coef_[0]
		gamma = 1.0 / (nfeatures)
		self.nSV = len(cls.support_vectors_)
		# Extract weight matrix
		self.pairWeights = cp.array( ((SV.T * c) @ SV) * gamma * gamma )
		# Extract Bias
		self.bias = -bias
	
	def getTrainer(self):
		return lambda ts: sequenceModelSVMOptimizedQuadraticCUDA(self.name, self.features, ts, windowSize = self.windowSize, windowStep = self.windowStep, alpha = self.alpha, labelPositive = self.labelPositive, labelNegative = self.labelNegative)
	
	def getSequenceFeatureVector(self, seq):
		return self.features.getAll(seq)
	
	def scoreWindow(self, seq):
		fv = cp.array(self.getSequenceFeatureVector(seq))
		return self.pairWeights @ fv @ fv - self.bias
	
	def __str__(self):
		return 'Support Vector Machine<Features: %s (%d); Training set: %s; Positive label: %s; Negative label; Kernel: %s; Support vectors: %d; Quadratic optimized, CUDA>'%(str(self.features), len(self.features), str(self.trainingSet), str(self.labelPositive), str(self.labelNegative), [ 'linear', 'quadratic', 'cubic' ][self.kernel-1], self.nSV)
	
	def __repr__(self): return self.__str__()

