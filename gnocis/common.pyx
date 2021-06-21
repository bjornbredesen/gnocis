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
from math import log


############################################################################
# General

# Sets the runtime seed
def setSeed(seed):
	"""
	Sets the random seed.
	
	:param seed: Sequence to get reverse complement of.
	:type seed: int
	"""
	random.seed(seed)

# Gets a float, with the same precision as used internally in Gnocis
def getFloat(float v):
	"""
	Helper function to get floating point values of the same precision as used for `float` by Cython.
	
	:param v: Value.
	:type v: float
	"""
	cdef float vr = v
	return vr

def mean(X):
	return sum(X) / len(X)

def std(X):
	xM = mean(X)
	return ( (1/(len(X) - 1)) * sum( ( x - xM )**2. for x in X ) )**0.5

def SE(X):
	return std(X) / (len(X)**0.5)

_CIFunc = lambda X: SE(X) * 1.96

def setConfidenceIntervalFunction(CIfunc):
	"""
	Sets the function for calculating confidence intervals.
	
	:param CIfunc: Function that takes a set of values and outputs the confidence interval difference from the mean. A symmetric confidence interval is assumed.
	:type CIfunc: function
	"""
	global _CIFunc
	_CIFunc = CIfunc

def useSciPyConfidenceIntervals(alpha = 0.95, useT = False):
	"""
	Sets the function for calculating confidence intervals.
	
	:param CIfunc: Function that takes a set of values and outputs the confidence interval difference from the mean. A symmetric confidence interval is assumed.
	:type CIfunc: function
	"""
	import scipy.stats as st
	def cif(X):
		m = mean(X)
		se = st.sem(X)
		if len(X) < 2 or se == 0.0: return 0.
		if useT: a, b = st.t.interval(alpha, len(X)-1, loc = m, scale = se)
		else: a, b = st.norm.interval(alpha, loc = m, scale = se)
		return b - m
	setConfidenceIntervalFunction(cif)

def CI(X):
	"""
	Calculates a confidence interval of the mean for a set of values. By default, Gnocis calculates a 95% confidence interval with a normal distribution. It is recommended to replace this with an appropriate distribution depending on the analysis performed.
	
	:param X: Values.
	:type X: list
	
	:return: Confidence interval difference from the mean.
	:rtype: float
	"""
	return _CIFunc(X)

def KLdiv(muA, varA, muB, varB):
	sigmaA, sigmaB = varA**0.5, varB**0.5
	if varB == 0. or sigmaA == 0.:
		return 0.
	return ( (muA-muB)**2. + varA - varB ) / (2.*varB) + log(sigmaB/sigmaA)


############################################################################
# Nucleic acids

"""
The set of nucleotides.
"""
nucleotides = [ 'A', 'T', 'G', 'C' ]

"""
A dictionary mapping nucleotides to their complements.
"""
complementaryNucleotides = { 'A':'T', 'T':'A', 'G':'C', 'C':'G', 'N':'N', 'a':'t', 't':'a', 'g':'c', 'c':'g', 'n':'n', ')':'(', '(':')', '|':'|' }

# Returns the reverse complement of a DNA sequence.
def getReverseComplementaryDNASequence(seq):
	"""
	Returns the reverse complement of a DNA sequence.
	
	:param seq: Sequence to get reverse complement of.
	:type seq: str
	"""
	return ''.join( complementaryNucleotides[nt] for nt in reversed(seq) )

"""
The set of IUPAC nucleotide codes (https://www.bioinformatics.org/sms/iupac.html).
"""
IUPACNucleotideCodes = [ 'R', 'Y', 'S', 'W', 'K', 'M', 'B', 'D', 'H', 'V', 'N' ]

"""
Maps each IUPAC nucleotide code (https://www.bioinformatics.org/sms/iupac.html) to a list of matching nucleotides.
"""
IUPACNucleotideCodeSemantics = {
	'R': ['A', 'G'],
	'Y': ['C', 'T'],
	'S': ['G', 'C'],
	'W': ['A', 'T'],
	'K': ['G', 'T'],
	'M': ['A', 'C'],
	'B': ['C', 'G', 'T'],
	'D': ['A', 'G', 'T'],
	'H': ['A', 'C', 'T'],
	'V': ['A', 'C', 'G'],
	'N': ['A', 'C', 'G', 'T'],
}

"""
Maps nucleotides to integers.
"""
kSpectrumNT2Index = { 'A':0, 'T':1, 'G':2, 'C':3, 'a':0, 't':1, 'g':2, 'c':3, 'N':-1, 'n':-1 }
"""
Maps integers to nucleotides.
"""
kSpectrumIndex2NT = { 0:'A', 1:'T', 2:'G', 3:'C', -1:'N' }


############################################################################
# Global data

PyMEAPrivateData = None

cdef class _PyMEAPrivateData:
	
	def __init__(self):
		cdef unsigned char* ltbl = <unsigned char*> malloc(256)
		cdef unsigned char bnt
		for i in range(256): ltbl[i] = <unsigned char>0xFF
		for i, nt in enumerate(nucleotides): ltbl[ord(nt)] = i
		self.charNTIndexTable = ltbl
	
	def __dealloc__(self):
		if self.charNTIndexTable:
			free(self.charNTIndexTable)
			self.charNTIndexTable = NULL

cdef _PyMEAPrivateData getGlobalData():
	global PyMEAPrivateData
	if not PyMEAPrivateData:
		PyMEAPrivateData = _PyMEAPrivateData()
	return PyMEAPrivateData

