# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from __future__ import division
from libcpp cimport bool
from .motifs cimport *
from .sequences cimport *


############################################################################
# Feature network

#---------------------
# Base class

cdef class featureNetworkNode:
	
	cpdef list get(self, sequence seq)

#---------------------
# Node type: Motif pair occurrence frequencies

cdef class FNNMotifPairOccurrenceFrequencies(featureNetworkNode):
	
	cdef public object motifs
	cdef public int distCut
	
	cpdef list get(self, sequence seq)
	

