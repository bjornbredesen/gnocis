# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from __future__ import division
from libcpp cimport bool
from .sequences cimport sequence, sequences


############################################################################
# Feature network

#---------------------
# Base class

cdef class featureNetworkNode:
	
	cpdef list get(self, sequences seq)

#---------------------
# Node type: Motif occurrence frequencies

cdef class FNNMotifOccurrenceFrequencies(featureNetworkNode):
	
	cdef public object motifs
	
	cpdef list getSeqVec(self, sequence seq)
	cpdef list get(self, sequences seq)

#---------------------
# Node type: Motif pair occurrence frequencies

cdef class FNNMotifPairOccurrenceFrequencies(featureNetworkNode):
	
	cdef public object motifs
	cdef public int distCut
	
	cpdef list getSeqVec(self, sequence seq)
	cpdef list get(self, sequences seq)

#---------------------
# Node type: k-spectrum

cdef class kSpectrum(featureNetworkNode):
	
	cdef public int nspectrum, nFeatures, bitmask
	cdef public dict kmerByIndex
	
	cpdef list getSeqVec(self, sequence seq)
	cpdef list get(self, sequences seq)

#---------------------
# Node type: k-spectrum mismatch

cdef class kSpectrumMM(featureNetworkNode):
	
	cdef public int nspectrum, nFeatures, bitmask
	cdef public dict kmerByIndex
	
	cpdef list getSeqVec(self, sequence seq)
	cpdef list get(self, sequences seq)
	
#---------------------
# Node type: Scaler

cdef class FNNScaler(featureNetworkNode):
	
	cdef public featureNetworkNode features
	cdef list vScale, vSub
	
	cpdef list get(self, sequences seq)
	
#---------------------
# Node type: Filter

cdef class FNNFilter(featureNetworkNode):
	
	cdef public featureNetworkNode features
	cdef list indices
	
	cpdef list get(self, sequences seq)

#---------------------
# Node type: Concatenation

cdef class FNNCat(featureNetworkNode):
	
	cdef public list inputs
	
	cpdef list get(self, sequences seq)

#---------------------
# Node type: Log-odds

cdef class FNNLogOdds(featureNetworkNode):

	cdef public featureNetworkNode features
	cdef public object labelPositive, labelNegative
	cdef public object trainingSet
	cdef public list _weights

	cpdef list get(self, sequences seq)


