# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

############################################################################
# Biomarker sets

# Represents biomarkers
cdef class biomarkers:
	
	cdef public str name
	cdef public int positiveThreshold, negativeThreshold
	cdef public dict biomarkers

