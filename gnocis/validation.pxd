# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from __future__ import division
from libcpp cimport bool


############################################################################
# Validation

# Represents a two-dimensional point.
cdef class point2D:
	
	cdef public float x, y

# Represents a pair of a validation score and label (with the option of a name).
cdef class validationPair:
	
	cdef public float score
	cdef public bool label
	cdef public str name

