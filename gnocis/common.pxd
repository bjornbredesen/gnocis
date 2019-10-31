# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

############################################################################
# Global data

cdef class _PyMEAPrivateData:
	
	cdef public unsigned char* charNTIndexTable

cdef _PyMEAPrivateData getGlobalData()

