# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from libcpp cimport bool
from .regions cimport *


############################################################################
# Sequence labels

# Represents a sequence label
cdef class sequenceLabel:
	
	cdef public str name
	cdef public float value


############################################################################
# Sequences

# Represents and loads DNA sequences.
cdef class sequence:
	
	cdef public str name, seq, path, annotation
	cdef public region sourceRegion
	cdef bytes cbytes, cbytesIndexed
	cdef public bytes getBytes(self)
	cdef public bytes getBytesIndexed(self)
	cdef public set labels

# Represents a set of DNA sequences.
cdef class sequences:
	
	cdef public str name
	cdef public list sequences


############################################################################
# Sequence streaming
# Allows for streaming of sequences from disk, and gradually processing them.
# Useful for extracting information from large sequences without storing them in memory, such as with entire genomes.

# Represents sequence streams.
cdef class sequenceStream:
	pass

# Streams a FASTA file in blocks.
cpdef streamFASTA(path, wantBlockSize = *, spacePrune = *, dropChr = *, restrictToSequences = *)

# Streams a 2bit file in blocks.
cpdef stream2bit(path, wantBlockSize = *, spacePrune = *, dropChr = *, restrictToSequences = *)

# Gets a sequence stream based on a path, deciding on the format from the path.
cpdef getSequenceStreamFromPath(path, wantBlockSize = *, spacePrune = *, dropChr = *, restrictToSequences = *)

###########################################################################
# Sequence generation

# Represents sequence streams
cdef class sequenceGenerator:
	pass

cdef class MarkovChain(sequenceGenerator):
	
	cdef public int degree, pseudoCounts, nGenerated
	cdef public bool addReverseComplements, prepared
	cdef public object trainingSequences
	cdef public list spectrum, initialDistribution
	cdef public list probspectrum
	cdef public dict comparableSpectrum

# IID sequence model
cdef class IID(sequenceGenerator):
	
	cdef public int pseudoCounts, nGenerated
	cdef public bool addComplements, prepared
	cdef public object trainingSequences
	cdef public dict spectrum
	cdef public list ntDistribution


