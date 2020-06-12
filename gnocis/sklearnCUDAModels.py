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
import cupy as cp
import numpy as np
from sklearn import svm
from .featurenetwork import baseModel

# GPU implementation
class CUDASVM(baseModel):
	
	def __init__(self, model = None, labelPositive = positive, labelNegative = negative, kDegree = 1, C = 4, batchsize = 1000):
		super().__init__(enableMultiprocessing = False, batchsize = batchsize)
		self.labelPositive, self.labelNegative = labelPositive, labelNegative
		self.labelNegative = labelNegative
		self.C, self.kDegree = C, kDegree
		self.mdl = model
		if model is not None:
			self.SV = cp.array(model.support_vectors_)
			self.cN = cp.array(model.dual_coef_[0])
	
	def __str__(self):
		return 'Support Vector Machine (CUDA)<C: %s; Kernel degree: %s; Positive label: %s; Negative label: %s>'%(str(self.C), str(self.kDegree), str(self.labelPositive), str(self.labelNegative))
	
	def score(self, featureVectors):
		c0 = self.mdl.coef0
		bias = self.mdl.intercept_[0]
		nfeatures = len(self.mdl.support_vectors_[0])
		gamma = 1.0 / (nfeatures)
		ret = []
		fvs = cp.array(featureVectors)
		rmat = ((gamma * (fvs @ self.SV.T) + c0) ** self.kDegree) @ self.cN + bias
		return [ [ v ] for v in rmat ]
		#return [ [ float(v) ] for v in rmat ]
	
	def train(self, trainingSet):
		fvPos = trainingSet[self.labelPositive]
		fvNeg = trainingSet[self.labelNegative]
		cP = [ 1.0 for _ in range(len(fvPos)) ]
		cN = [ -1.0 for _ in range(len(fvNeg)) ]
		mdl = svm.SVC(C = self.C, kernel = 'poly', degree = self.kDegree, gamma = 'auto')
		mdl.fit( np.array(fvPos+fvNeg), np.array(cP+cN) )
		return CUDASVM(
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

