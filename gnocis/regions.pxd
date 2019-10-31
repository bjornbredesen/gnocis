# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from libcpp cimport bool


############################################################################
# Regions

# Represents regions.
cdef class region:
	
	cdef public str seq
	cdef public long long start, end
	cdef public bool strand
	cdef public float score
	cdef public str source, feature, group
	cdef public ext
	cdef public set markers

# Represents region sets. Supports operands '+' for adding regions from two sets with no merging, '|' for merging two sets, and '^' for excluding the regions in the latter set from the first.
cdef class regions:
	
	cdef public str name
	cdef public list regions

