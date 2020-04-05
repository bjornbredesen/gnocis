# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from __future__ import division
import random
from libcpp cimport bool
from libc.stdlib cimport malloc, free	
from libc.string cimport memcpy


############################################################################
# Validation

# Represents a two-dimensional point.
cdef class point2D:
	"""
	Two-dimensional point.
	
	:param x: X-coordinate.
	:param y: Y-coordinate.
	:param rank: Rank.
	
	:type x: float
	:type y: float
	:type rank: float, optional
	"""
	
	def __init__(self, x, y, rank = 0.):
		self.x, self.y, self.rank = x, y, rank

# Represents a pair of a validation score and label (with the option of a name).
cdef class validationPair:
	"""
	Pair of score and binary class label.
	
	:param score: Score.
	:param label: Binary class label.
	
	:type score: float
	:type float: bool
	"""
	
	def __init__(self, score, label, name = ''):
		self.score = score
		self.label = label
		self.name = name
	
	def __str__(self):
		return '(%.2f, %s)'%(self.score, '+' if self.label else '-')
	
	def __repr__(self): return self.__str__()
	

# Gets the Receiver Operating Characteristic for sets of positive and negative validation pairs
def getROC(vPos, vNeg):
	"""
	Generates a Receiver Operating Characteristic (ROC) curve.
	
	:param vPos: Positive validation pairs.
	:param vNeg: Negative validation pairs.
	:type vPos: list
	:type vNeg: list
	
	:return: List of 2D-points for curve
	:rtype: list
	"""
	if len(vPos) == 0 or len(vNeg) == 0:
		return [ point2D(0.0, 0.0), point2D(1.0, 1.0) ]
	vPairs = sorted(vPos + vNeg, key = lambda x: -x.score)
	TP, FP = 0, 0
	rank = 0
	curve = [ point2D(0.0, 0.0) ]
	for vp in vPairs:
		if vp.label:
			TP += 1
		else:
			FP += 1
		rank += 1
		curve .append(point2D( x = FP / len(vNeg), y = TP / len(vPos), rank = rank ))
	curve.append(point2D(1.0, 1.0))
	return curve

# Gets the Precision/Recall Curve for sets of positive and negative validation pairs
def getPRC(vPos, vNeg, subdivisions = 4):
	"""
	Generates a Precision/Recall Curve (PRC).
	
	:param vPos: Positive validation pairs.
	:param vNeg: Negative validation pairs.
	:type vPos: list
	:type vNeg: list
	
	:return: List of 2D-points for curve
	:rtype: list
	"""
	if len(vPos) == 0 or len(vNeg) == 0:
		return [ point2D(0.0, 0.0), point2D(1.0, 0.0) ]
	vPairs = sorted(vPos + vNeg, key = lambda x: -x.score)
	TP, FP = 0., 0.
	rank = 0.
	curve = [ point2D(0.0, 0.0) ]
	for vp in vPairs:
		for sdv in range(subdivisions):
			if vp.label:
				TP += 1. / subdivisions
			else:
				FP += 1. / subdivisions
			rank += 1. / subdivisions
			curve.append(point2D( x = TP / len(vPos), y = TP / (TP + FP), rank = rank ))
	curve.append(point2D( x = 1.0, y = len(vPos)/(len(vPos)+len(vNeg)) ))
	return curve

# Returns the area under a given curve. Used for calculation of threshold-less statistics.
def getAUC(curve):
	"""
	Calculates the Area Under the Curve (AUC) for an input curve.
	
	:param curve: Curve to calculate area under.
	:type curve: list
	
	:return: Area Under the Curve (AUC).
	:rtype: float
	"""
	return sum( (b.x - a.x) * (b.y + a.y) * 0.5 for a, b in zip(curve[:-1], curve[1:]) )

# Gets a confusion matrix for sets of positive and negative class validation pairs.
def getConfusionMatrix(vPos, vNeg, threshold = 0.0):
	"""
	Calculates confusion matrix based on input validation pairs, and an optional threshold.
	
	:param vPos: Positive validation pairs.
	:param vNeg: Negative validation pairs.
	:param threshold: Classification threshold.
	:type vPos: list
	:type vNeg: list
	:type threshold: float
	
	:return: Dictionary with confusion matrix entries.
	:rtype: dict
	"""
	TP, FP, TN, FN = 0, 0, 0, 0
	for vp in vPos:
		if vp.score > threshold:
			TP += 1
		else:
			FN += 1
	for vp in vNeg:
		if vp.score > threshold:
			FP += 1
		else:
			TN += 1
	return { 'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN }

# Returns a dictionary of common statistics for a confusion matrix dictionary.
def getConfusionMatrixStatistics(CM):
	"""
	Calculates confusion matrix statistics in an input confusion matrix dictionary.
	
	:param CM: Confusion matrix dictionary.
	:type CM: dict
	
	:return: Dictionary with confusion matrix statistics.
	:rtype: dict
	"""
	nPos = CM['TP'] + CM['FN']
	nNeg = CM['FP'] + CM['TN']
	MCCFac = (CM['TP']+CM['FP'])*(CM['TP']+CM['FN'])*(CM['TN']+CM['FP'])*(CM['TN']+CM['FN'])
	return {
		'TP': CM['TP'], 'FP': CM['FP'], 'TN': CM['TN'], 'FN': CM['FN'],
		'Accuracy': ((CM['TP'] + CM['TN']) / (nPos + nNeg)) if (nPos + nNeg) > 0 else 0.0,
		'MCC': ((CM['TP']*CM['TN'] - CM['FP']*CM['FN']) / (MCCFac**0.5)) if MCCFac != 0.0 else 0.0,
		'Recall': (CM['TP'] / nPos) if nPos > 0 else 0.0,
		'Precision': (CM['TP'] / (CM['TP'] + CM['FP'])) if (CM['TP'] + CM['FP']) > 0 else 0.0,
	}

# Prints validation statistics.
def printValidationStatistics(stats):
	"""
	Prints model confusion matrix statistics from a dictionary.
	
	:param stats: Confusion matrix statistics dictionary.
	:type stats: dict
	"""
	print(stats['title'])
	print(' - Model: %s'%(stats['model']))
	print(' - Positives: %s'%str(stats['positives']))
	print(' - Negatives: %s'%str(stats['negatives']))
	print(' - Stats')
	print(' -  - TP: %5d    - FP: %5d'%(stats['TP'], stats['FP']))
	print(' -  - FN: %5d    - TN: %5d'%(stats['FN'], stats['TN']))
	for stat in [ 'Accuracy', 'MCC', 'Recall', 'Precision', 'ROC AUC', 'PRC AUC' ]:
		print(' -  - %-10s %6.2f %%'%(stat + ':', 100.0 * stats[stat]))

