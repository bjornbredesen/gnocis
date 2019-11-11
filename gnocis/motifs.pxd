# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from __future__ import division
import re
from libcpp cimport bool
from libc.stdlib cimport malloc, free	
from libc.string cimport memcpy
from .sequences cimport *


############################################################################
# Motifs

# Represents a DNA sequence motif occurrence.
cdef class motifOccurrence:
	
	cdef public object motif
	cdef public object seq
	cdef public int start, end
	cdef public bool strand


############################################################################
# IUPAC motifs

cdef class IUPACMotif:
	
	cdef public str name, motif, regexMotif, regexMotifRC
	cdef public int nmismatches
	cdef public object c, cRC
	cdef public sequence cachedSequence
	cdef public list cachedOcc


############################################################################
# Position Weight Matrix motifs

cdef class PWMMotif:
	
	cdef public str name, path
	cdef public double threshold
	cdef public list PWMF, PWMRC
	cdef double* bPWMF
	cdef double* bPWMRC
	cdef double* maxScoreLeftF
	cdef double* maxScoreLeftRC
	cdef public sequence cachedSequence
	cdef public list cachedOcc

