# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from __future__ import division
import re
import struct
import random
import gzip
from libcpp cimport bool
from libc.stdlib cimport malloc, free	
from libc.string cimport memcpy
from .common cimport _PyMEAPrivateData, getGlobalData
from .common import nucleotides, complementaryNucleotides, getReverseComplementaryDNASequence, kSpectrumNT2Index, kSpectrumIndex2NT
from .regions cimport *
from .motifs cimport *


############################################################################
# Sequence labels

# Represents a sequence label
cdef class sequenceLabel:
	"""
	The `sequenceLabel` class represents a label that can be assigned to sequences to be used for model training or testing, such as "positive" or "negative".
	
	:param name: Name of the sequence label.
	:param value: Value to represent the label.
	
	:type name: str
	:type value: float
	"""
	
	def __init__(self, name, value):
		self.name, self.value = name, value
	
	def __str__(self):
		return 'Sequence label<%s; value = %s>'%(self.name, str(self.value))
	
	def __eq__(self, o):
		return self.value == o.value
	
	def __hash__(self):
		return hash(self.value)
	
	def __repr__(self):
		return self.__str__()

positive = sequenceLabel('Positive', 1.0)
negative = sequenceLabel('Negative', -1.0)


###########################################################################
# Sequences

# Represents and loads DNA sequences.
cdef class sequence:
	"""
	The `sequence` class represents a DNA sequence. 
	
	:param name: Name of the sequence.
	:param seq: Sequence.
	:param path: Source file path.
	:param sourceRegion: Source region.
	:param annotation: Annotation.
	
	:type name: str
	:type seq: str
	:type path: str, optional
	:type sourceRegion: region, optional
	:type annotation: str, optional
	
	For a `sequence` instance `S`, `len(S)` gives the sequence length. `[ nt for nt in S ]` lists the nucleotides. For two sequences `A` and `B`, `A+B` gives the concatenated sequence.
	"""
	
	#__slots__ = 'name', 'seq', 'path'
	
	def __init__(self, name, seq, path = '', sourceRegion = None, annotation = None, labels = None):
		self.name, self.seq, self.path = name, seq, path
		self.sourceRegion = sourceRegion
		self.annotation = annotation
		self.labels = set() if labels is None else labels
		if name == None and sourceRegion != None:
			self.name = '%s:%d..%d (%s)'%(sourceRegion.seq, sourceRegion.start, sourceRegion.end, '+' if sourceRegion.strand else '-')
	
	def __eq__(self, other):
		return self.seq == other.seq
	
	def __hash__(self):
		return hash(self.seq)
	
	def __len__(self):
		return len(self.seq)
	
	def __str__(self):
		return 'Sequence<%s>'%(self.name + (' (%s)'%self.annotation if self.annotation != None else ''))
	
	def __repr__(self):
		return self.__str__()
	
	def label(self, label):
		"""
		:param label: Label to add.
		:type label: sequenceLabel
		
		:return: Returns a labelled sequence.
		:rtype: sequence
		"""
		return sequence(name = self.name, seq = self.seq, path = self.path, sourceRegion = self.sourceRegion, annotation = self.annotation, labels = self.labels | set([label]))
	
	# Extracts sequence windows with a desired size and step size.
	def windows(self, int size, int step, bool includeCroppedEnds = False):
		"""
		:param size: Window size.
		:param step: Window step size.
		:param includeCroppedEnds: Whether or not to include cropped windows on ends of sequence. Default is `False`.
		:type size: int
		:type step: int
		:type includeCroppedEnds: bool, optional
		
		:return: Returns a sequence set of sliding window sequences over the sequence.
		:rtype: sequences
		"""
		cdef list windows
		cdef sequences wsequences
		cdef long long wA, wB
		windows = []
		wA = 0
		while True:
			wB = wA + size
			if wB > len(self.seq):
				if includeCroppedEnds:
					windows.append( sequence('%s:%d..%d'%(self.name, wA, len(self.seq)), self.seq[wA:], self.path, sourceRegion = region(self.name, wA, len(self.seq)), labels = self.labels) )
				break
			windows.append( sequence('%s:%d..%d'%(self.name, wA, wB), self.seq[wA:wB], self.path, sourceRegion = region(self.name, wA, wB), labels = self.labels) )
			wA += step
		wsequences = sequences('%s windows(%d/%d)'%(self.name, size, step), windows)
		return wsequences
	
	cdef public bytes getBytes(self):
		"""
		:return: Returns the bytes of the sequence.
		:rtype: bytes
		"""
		if not self.cbytes:
			self.cbytes = <bytes> self.seq.encode('UTF-8')
		return self.cbytes
	
	cdef public bytes getBytesIndexed(self):
		"""
		:return: Returns the indexed bytes of the sequence.
		:rtype: bytes
		"""
		cdef bytes cbytes
		cdef _PyMEAPrivateData gdat
		cdef unsigned char* ltbl
		if not self.cbytesIndexed:
			cbytes = self.getBytes()
			gdat = getGlobalData()
			ltbl = gdat.charNTIndexTable
			self.cbytesIndexed = bytes([ ltbl[cbytes[i]] for i in range(len(self.seq)) ])
		return self.cbytesIndexed

# Represents a set of DNA sequences.
cdef class sequences:
	"""
	The `sequences` class represents a set of DNA sequences. 
	
	:param name: Name of the sequence set.
	:param seq: List of sequences to include.
	
	:type name: str
	:type seq: list
	
	For a `sequence` instance `S`, `len(S)` gives the sequence length, and `[ s for s in S ]` lists the sequences. For two sequence sets `A` and `B`, `A+B` gives the set containing all sequences from both sets.
	"""
	
	#__slots__ = 'name', 'sequences'
	
	def __init__(self, name, seq):
		self.name = name
		self.sequences = seq
	
	def __iter__(self):
		return self.sequences.__iter__()
	
	def __getitem__(self, i):
		return self.sequences[i]
	
	def __len__(self):
		return len(self.sequences)
	
	def __str__(self):
		return 'Sequence list<%s>'%(self.name)
	
	def __repr__(self):
		return self.__str__()
	
	def __add__(self, other):
		return sequences('%s + %s'%(self.name, other.name), self.sequences + other.sequences)
	
	# Returns n split sequence sets
	def split(self, n):
		"""
		:param n: Number of sets to split into.
		:type n: int
		
		:return: Returns a list of `n` independent splits of the sequence set, without shuffling.
		:rtype: list
		"""
		cA = 0
		ret = []
		for i in range(n):
			cB = int( ( len(self.sequences) * (i+1) / n ) )
			if cA == cB: continue
			ret.append( sequences(self.name + ' [%d:%d]'%(cA, cB), self.sequences[ cA : cB ]) )
			cA = cB
		return ret
	
	# Returns a random division of the sequences
	def randomSplit(self, ratio = 0.5):
		"""
		:param ratio: Ratio for the split, between 0 and 1. For the return touple, `(A, B)`, a ratio of 0 means all the sequences are in `B`, and opposite for a ratio of 1.
		:type ratio: float
		
		:return: Returns a random split of the sequences into two independent sets, with the given ratio.
		:rtype: sequences
		"""
		seqs = self.sequences[:]
		isplit = int(len(seqs)*ratio)
		random.shuffle(seqs)
		return sequences(self.name + ' (split A)', seqs[:isplit]),\
		       sequences(self.name + ' (split B)', seqs[isplit:])
	
	# Returns a renamed sequence set
	def rename(self, newname):
		"""
		:param newname: New name.
		:type newname: str
		
		:return: Renamed sequence set.
		:rtype: sequences
		"""
		return sequences(newname, self.sequences)
	
	# Gets a sequence set with windows from the sequences
	def windows(self, size, step):
		""" Generates and returns a set of sliding window sequences.
		
		:param size: Window size.
		:param step: Window step size.
		:type size: int
		:type step: int
		
		:return: Sequence set of all sliding window sequences across all sequences in the set.
		:rtype: sequences
		"""
		return sequences(self.name + ' (windows - size: %d bp; step: %d bp)'%(size, step), list(streamSequenceWindows( self, size, step )))
	
	# Gets a region set with windows from the sequences
	def windowRegions(self, size, step):
		""" Generates and returns a set of sliding window regions.
		
		:param size: Window size.
		:param step: Window step size.
		:type size: int
		:type step: int
		
		:return: Region set of all sliding window regions across all sequences in the set.
		:rtype: regions
		"""
		seqLengths = self.sequenceLengths()
		rset = []
		for cseq in sorted(seqLengths.keys()):
			cstart = 0
			while cstart + size <= seqLengths[cseq]:
				rset.append( region(cseq, cstart, cstart+size-1) )
				cstart += step
		return regions('Windows (Window size: %d bp; step size: %d bp)'%(size, step), rset, initialSort = False)
	
	# Gets the regions of origin for the sequences (if set)
	def sourceRegions(self):
		"""
		:return: Region set of source regions for all sequences in the set.
		:rtype: regions
		"""
		return regions(self.name, [ s.sourceRegion for s in self.sequences if s.sourceRegion != None ])
	
	# Adds a label to sequences
	def label(self, label):
		""" Adds a label to sequences.
		
		:param label: Label to add to sequences.
		:type label: sequenceLabel
		
		:return: Sequences with label added.
		:rtype: sequences
		"""
		return sequences(self.name, [
			s.label(label) for s in self.sequences
		])
	
	# Gets sequences with the given label
	def withLabel(self, labels, ensureAllLabels = True):
		""" Extracts sequences with the given label or labels.
		
		:param labels: List of labels, or single labels, to extract sequences with.
		:param ensureAllLabels: If true, an exception is raised if no sequences with a label were found. Default true.
		:type labels: list, sequenceLabel
		:type ensureAllLabels: bool
		
		:return: Single label or list of labels.
		:rtype: sequences, list
		
		If multiple labels are given, a list of `sequences` instances are returned in the same order as the labels.
		"""
		if isinstance(labels, list):
			ret = [
				sequences(self.name + ' (%s)'%lbl.name, [
					s for s in self.sequences if lbl in s.labels
				])
				for lbl in labels
			]
			if ensureAllLabels and any(len(seqs) == 0 for seqs in ret):
				raise Exception("Not all requested labels were found among the sequences")
			return ret
		else:
			ret = sequences(self.name + ' (%s)'%labels.name, [
				s for s in self.sequences if labels in s.labels
			])
			if ensureAllLabels and len(ret) == 0:
				raise Exception("No sequences with the requested label were found")
			return ret
	
	# Gets labels for sequences
	def labels(self):
		""" Gets labels for sequences.
		
		:return: Set of labels.
		:rtype: set
		"""
		return set(l for s in self.sequences for l in s.labels)
	
	# Saves to FASTA file.
	def saveFASTA(self, path):
		""" Saves the regions to a FASTA file.
		
		:param path: Output file path.
		:type path: str
		"""
		with open(path, 'w') as fOut:
			for seq in self.sequences:
				fOut.write(">%s\n%s\n"%(seq.name.replace(' ', '_'), re.sub("(.{65})","\\1\n", seq.seq, 0, re.DOTALL)))
	
	# Gets the sequence lengths
	def sequenceLengths(self):
		"""
		:return: Returns a dictionary of the lengths of all sequences in the set.
		:rtype: dict
		"""
		cdef dict ret
		cdef object cseq, blk
		return { cseq.name: len(cseq) for cseq in self }
	
	# Prints basic statistics about the set.
	def printStatistics(self):
		"""
		Outputs basic statistics.
		"""
		print('Sequence set statistics - ' + self.name)
		print(' - Sequences: %d'%len(self.sequences))
		if len(self.sequences) > 0:
			print(' - Mean length: %.2f nt (min.: %d nt - max.: %d nt)'%( sum( len(s) for s in self.sequences ) / len(self.sequences), min( len(s) for s in self.sequences), max( len(s) for s in self.sequences) ))
		print(' - Total length: %d nt'%( sum( len(s) for s in self.sequences ) ))

# Loads a FASTA sequence file and returns a list of sequences contained in the file.
def loadFASTA(path):
	""" Loads sequences from a FASTA file.
	
	:param path: Path to the input file.
	:type path: str
	
	:return: Loaded sequences.
	:rtype: sequences
	"""
	lst = []
	pathName = path.split('/')[-1]
	with open(path, 'r') as f:
		for line in f:
			line = line.strip()
			if line.startswith('>'):
				lst.append(sequence(line[1:], '', path, annotation = 'From FASTA file ' + pathName))
			elif len(lst) == 0:
				raise Exception('FASTA sequence error')
			else:
				lst[-1].seq += line
	return sequences('FASTA file: ' + pathName, lst)

# Loads a FASTA sequence file and returns a list of sequences contained in the file.
def loadFASTAGZ(path):
	""" Loads sequences from a gzipped FASTA file.
	
	:param path: Path to the input file.
	:type path: str
	
	:return: Loaded sequences.
	:rtype: sequences
	"""
	lst = []
	pathName = path.split('/')[-1]
	with gzip.open(path, 'rb') as f:
		for line in (y.decode('utf-8') for y in f):
			line = line.strip()
			if line.startswith('>'):
				lst.append(sequence(line[1:], '', path, annotation = 'From FASTA file ' + pathName))
			elif len(lst) == 0:
				raise Exception('FASTA sequence error')
			else:
				lst[-1].seq += line
	return sequences('FASTA file: ' + pathName, lst)


############################################################################
# Sequence streaming
# Allows for streaming of sequences from disk, and gradually processing them.
# Useful for extracting information from large sequences without storing them in memory, such as with entire genomes.

# Represents sequence streams.
cdef class sequenceStream:
	"""
	The `sequenceStream` class represents sequence streams. 
	"""
	
	def __init__(self):
		pass
	
	# Fetches n sequences (or less, if fewer or none are left)
	def fetch(self, n, maxnt):
		"""
		Fetches n sequences (or less, if fewer or none are left).
		
		:param n: Number of sequences to fetch.
		:param maxnt: Maximum number of nucleotides to fetch.
		:type n: int
		:type maxnt: int
		
		:return Yields sequences.
		:rtype: sequences (yield)
		"""
		cdef list fetched
		cdef int i
		cdef long long ntFetched
		cdef sequence cseq
		fetched = []
		i = 0
		ntFetched = 0
		for cseq in self.streamFullSequences():
			fetched.append(cseq)
			i += 1
			ntFetched += len(fetched[-1])
			#if i >= n or ntFetched >= maxnt:
			if ntFetched >= maxnt:
				i = 0
				ntFetched = 0
				yield sequences(self.name, fetched)
				fetched = []
		if len(fetched) > 0:
			yield sequences(self.name, fetched)
	
	# Outputs a stream of complete sequences by joining sequence blocks
	def streamFullSequences(self):
		"""
		Streams and yields whole sequences.
		
		:return Yields whole sequence.
		:rtype: sequence (yield)
		"""
		for cseq in self:
			yield sequence(cseq.name, ''.join(blk for blk in cseq))

	# Gets the sequence lengths
	def sequenceLengths(self):
		"""
		:return: Returns a dictionary of the lengths of all sequences in the stream.
		:rtype: dict
		"""
		cdef dict ret
		cdef object cseq, blk
		return { cseq.name: sum( len(blk) for blk in cseq ) for cseq in self }
	
	# Gets a sequence set with windows from the sequences
	def windows(self, size, step):
		""" Generates and returns a set of sliding window sequences.
		
		:param size: Window size.
		:param step: Window step size.
		:type size: int
		:type step: int
		
		:return: Sequence set of all sliding window sequences across all sequences in the stream.
		:rtype: sequences
		"""
		return sequences(self.name + ' (windows - size: %d bp; step: %d bp)'%(size, step), list(streamSequenceWindows( self, size, step )))
	
	# Gets a region set with windows from the sequences
	def windowRegions(self, size, step):
		""" Generates and returns a set of sliding window regions.
		
		:param size: Window size.
		:param step: Window step size.
		:type size: int
		:type step: int
		
		:return: Region set of all sliding window regions across all sequences in the stream.
		:rtype: regions
		"""
		seqLengths = self.sequenceLengths()
		rset = []
		for cseq in sorted(seqLengths.keys()):
			cstart = 0
			while cstart + size <= seqLengths[cseq]:
				rset.append( region(cseq, cstart, cstart+size-1) )
				cstart += step
		return regions('Windows (Window size: %d bp; step size: %d bp)'%(size, step), rset, initialSort = False)

# Represents a sequence in a FASTA file
cdef class streamFASTASeq:
	
	cdef public str name, lastLine
	cdef public int wantBlockSize
	cdef public bool parsingCompleted
	cdef public object fstream
	
	def __init__(self, name, fstream, wantBlockSize):
		self.name, self.fstream = name, fstream
		self.lastLine = None
		self.wantBlockSize = wantBlockSize
		self.parsingCompleted = False
	
	def __iter__(self):
		return self.stream()
	
	def __str__(self):
		return 'FASTA sequence stream<%s>'%(self.name)
	
	def __repr__(self):
		return self.__str__()
	
	def stream(self):
		cdef str cbuf, cl
		cdef int wbs = self.wantBlockSize
		cbuf = ''
		for cl in self.fstream:
			if cl.startswith('>'):
				# Output all that remains in buffer
				while len(cbuf) > 0:
					yield cbuf[:wbs]
					cbuf = cbuf[wbs:]
				# Store last line for continued parsing
				self.lastLine = cl
				self.parsingCompleted = True
				return None
			cbuf += cl.strip()
			# Output blocks of requested size from buffer
			while len(cbuf) >= wbs:
				yield cbuf[:wbs]
				cbuf = cbuf[wbs:]
		# Output all that remains in buffer
		while len(cbuf) > 0:
			yield cbuf[:wbs]
			cbuf = cbuf[wbs:]
		self.lastLine = None
		self.parsingCompleted = True

# Represents stream of FASTA sequences.
class sequenceStreamFASTA(sequenceStream):
	"""
	Streams a FASTA file.
	
	:param path: Path to the input file.
	:param wantBlockSize: Desired block size.
	:param spacePrune: Whether or not to prune sequence name spaces.
	:param dropChr: Whether or not to prune sequence name "chr"-prefixes.
	:param restrictToSequences: List of sequence names to restrict to/focus on.
	:param isGZ: Whether or not the input file is GZipped.
	
	:type path: str
	:type wantBlockSize: int
	:type spacePrune: bool
	:type dropChr: bool
	:type restrictToSequences: list
	:type isGZ: bool
	"""
	
	def __init__(self, path, wantBlockSize, spacePrune, dropChr, restrictToSequences, isGZ = False):
		self.path = path
		self.name = path.split('/')[-1]
		self.wantBlockSize = wantBlockSize
		self.spacePrune = spacePrune
		self.dropChr = dropChr
		self.restrictToSequences = restrictToSequences
		self.isGZ = isGZ
	
	def __iter__(self):
		cdef str line, sname
		cdef streamFASTASeq seqStream
		cdef object fIn, lineSrc
		with gzip.open(self.path, 'rb') if self.isGZ else open(self.path, 'r') as fIn:
			# Parse line by line, with streaming of all included sequences, re-using the last
			# read line by each sequence stream when a new sequence starts.
			lineSrc = (y.decode('utf-8') for y in fIn) if self.isGZ else fIn
			line = None
			while True:
				if not line:
					line = next(lineSrc)
				if line.startswith('>'):
					sname = line[1:].strip()
					if self.spacePrune:
						sname = sname.split(' ')[0]
					if self.dropChr and sname.lower().startswith('chr'):
						sname = sname[3:]
					if self.restrictToSequences != None and not sname in self.restrictToSequences:
						line = ''
						while line != None and not line.startswith('>'):
							line = next(lineSrc)
							if line == None:
								return None
						continue
					seqStream = streamFASTASeq(sname, lineSrc, self.wantBlockSize)
					yield seqStream
					if seqStream.parsingCompleted:
						line = seqStream.lastLine
						if line == None:
							return None
					else:
						line = ''
						while line != None and not line.startswith('>'):
							line = next(lineSrc)
							if line == None:
								return None
				else:
					raise Exception('FASTA file syntax error')
		return None
	
	def __str__(self):
		return 'FASTA sequence stream<%s>'%(self.name)
	
	def __repr__(self):
		return self.__str__()

# Streams a FASTA file in blocks.
cpdef streamFASTA(path, wantBlockSize = 5000, spacePrune = True, dropChr = True, restrictToSequences = None):
	"""
	Streams a FASTA file.
	
	:param path: Path to the input file.
	:param wantBlockSize: Desired block size.
	:param spacePrune: Whether or not to prune sequence name spaces.
	:param dropChr: Whether or not to prune sequence name "chr"-prefixes.
	:param restrictToSequences: List of sequence names to restrict to/focus on.
	
	:type path: str
	:type wantBlockSize: int, optional
	:type spacePrune: bool, optional
	:type dropChr: bool, optional
	:type restrictToSequences: list, optional
	
	:return: Generated sequence stream.
	:rtype: sequenceStream
	"""
	return sequenceStreamFASTA(path, wantBlockSize, spacePrune, dropChr, restrictToSequences)

# Streams a FASTA file in blocks.
cpdef streamFASTAGZ(path, wantBlockSize = 5000, spacePrune = True, dropChr = True, restrictToSequences = None):
	"""
	Streams a gzipped FASTA file.
	
	:param path: Path to the input file.
	:param wantBlockSize: Desired block size.
	:param spacePrune: Whether or not to prune sequence name spaces.
	:param dropChr: Whether or not to prune sequence name "chr"-prefixes.
	:param restrictToSequences: List of sequence names to restrict to/focus on.
	
	:type path: str
	:type wantBlockSize: int, optional
	:type spacePrune: bool, optional
	:type dropChr: bool, optional
	:type restrictToSequences: list, optional
	
	:return: Generated sequence stream.
	:rtype: sequenceStream
	"""
	return sequenceStreamFASTA(path, wantBlockSize, spacePrune, dropChr, restrictToSequences, isGZ = True)

# Represents a 2bit N-block
cdef class stream2bitNBlock:
	
	cdef public int start, size
	
	def __init__(self, start, size):
		self.start, self.size = start, size

# Represents a sequence in a 2bit file
cdef class stream2bitSeq:
	
	cdef public str name
	cdef public object fstream
	cdef public int wantBlockSize
	cdef public int nnt
	cdef public list Nblocks
	
	def __init__(self, name, fstream, wantBlockSize, Nblocks, nnt):
		self.name, self.fstream = name, fstream
		self.wantBlockSize = wantBlockSize
		self.Nblocks = Nblocks
		self.nnt = nnt
	
	def stream(self):
		cdef list Nblocks = self.Nblocks
		cdef stream2bitNBlock block
		# Make conversion table that converts a byte to four nucleotides at a time.
		# http://genome.ucsc.edu/FAQ/FAQformat.html#format7
		# T - 00, C - 01, A - 10, G - 11
		ntByIndex = [ <unsigned char>b'T', <unsigned char>b'C', <unsigned char>b'A', <unsigned char>b'G' ]
		cdef char* byteToNTs = <char*>malloc(256*4)
		for i in range(256):
			byteToNTs[i*4+0] = ntByIndex[(i>>6)&3]
			byteToNTs[i*4+1] = ntByIndex[(i>>4)&3]
			byteToNTs[i*4+2] = ntByIndex[(i>>2)&3]
			byteToNTs[i*4+3] = ntByIndex[i&3]
		cdef int* byteToNTsI = <int*> byteToNTs
		#
		cdef int nnt = self.nnt
		cdef int wantBlockSize = self.wantBlockSize
		cdef int nLoad = int((wantBlockSize+3)/4)
		cdef int nLoadNT = nLoad*4
		cdef int rcursor = 0
		cdef int filecursor = 0
		cdef int ints = 0
		cdef int rbufsize = nLoadNT
		cdef char* rbuf = <char*> malloc(rbufsize+4)
		cdef int* crbuf
		cdef int nprocess
		# FIle reading butter
		cdef bytes cbuf
		cdef unsigned char c
		#
		cdef bool isEnd = False
		while True:
			# Convert a byte to four nucleotides at a time using lookup table
			if nnt - filecursor < nLoadNT:
				nLoad = (nnt - filecursor + 3) >> 2
				nLoadNT = nLoad << 2
				isEnd = True
			cbuf = self.fstream.read(nLoad)
			crbuf = <int*>&rbuf[rcursor]
			for c in cbuf:
				crbuf[0] = byteToNTsI[c]
				crbuf += 1
			if isEnd:
				rcursor += nnt - filecursor
			else:
				rcursor += 4*len(cbuf)
			if rcursor >= rbufsize+4:
				raise Exception('WARNING! rcursor >= rbufsize+1! rcursor = %d'%rcursor)
			filecursor += nLoadNT
			# Output fragments of requested size
			while rcursor > 0:
				if rcursor < wantBlockSize and not isEnd:
					break
				nprocess = wantBlockSize
				if rcursor < nprocess:
					nprocess = rcursor
				firstValidNBlock = 0
				for iblock, block in enumerate(Nblocks):
					if block.start + block.size < ints:
						firstValidNBlock = iblock + 1
						continue
					if block.start >= ints + nLoadNT:
						break
					iA = max(block.start - ints, 0)
					iB = min(block.start + block.size - ints, nLoadNT)
					for x in range(iA, iB):
						rbuf[x] = <char>b'N'
				if firstValidNBlock > 0:
					Nblocks = Nblocks[firstValidNBlock:]
				ints += nprocess
				yield (<bytes>rbuf[:nprocess]).decode('UTF-8')
				memcpy(rbuf, &rbuf[nprocess], rbufsize-nprocess)
				rcursor -= nprocess
			if isEnd:
				break
		free(rbuf)
		free(byteToNTs)
	
	def __iter__(self):
		return self.stream()
	
	def __str__(self):
		return '2bit sequence stream<%s>'%(self.name)
	
	def __repr__(self):
		return self.__str__()

# Represents stream of 2bit sequences.
class sequenceStream2bit(sequenceStream):
	"""
	Streams a 2bit file.
	
	:param path: Path to the input file.
	:param wantBlockSize: Desired block size.
	:param spacePrune: Whether or not to prune sequence name spaces.
	:param dropChr: Whether or not to prune sequence name "chr"-prefixes.
	:param restrictToSequences: List of sequence names to restrict to/focus on.
	
	:type path: str
	:type wantBlockSize: int
	:type spacePrune: bool
	:type dropChr: bool
	:type restrictToSequences: list
	"""
	
	def __init__(self, path, wantBlockSize, spacePrune, dropChr, restrictToSequences):
		self.path = path
		self.name = path.split('/')[-1]
		self.wantBlockSize = wantBlockSize
		self.spacePrune = spacePrune
		self.dropChr = dropChr
		self.restrictToSequences = restrictToSequences
	
	def __iter__(self):
		cdef unsigned int sig, ver, nseq, reserved, offset
		cdef list seqs
		cdef int iseq
		cdef dict seqInfo
		cdef str sname, name
		cdef bytes namebytes
		cdef int nnt, nblk
		cdef list Nblocks, blkStarts, blkSizes, maskStarts, maskSizes
		# File processing
		with open(self.path, 'rb') as fIn:
			sig, ver, nseq, reserved = struct.unpack('IIII', fIn.read(16))
			if sig != 0x1A412743:
				raise Exception('Invalid 2bit sequence file. Header signature was of unexpected value.')
			seqs = []
			for iseq in range(nseq):
				namelen = struct.unpack('B', fIn.read(1))[0]
				namebytes = fIn.read(namelen)
				name = namebytes.decode("utf-8")
				if name.lower().startswith('chr'): name = name[3:]
				offset = struct.unpack('I', fIn.read(4))[0] # Relative to file start
				seqs.append(  { 'name': name, 'offset': offset } )
			for iseq, seqInfo in enumerate(seqs):
				sname = seqInfo['name']
				if self.spacePrune:
					sname = sname.split(' ')[0]
				if self.dropChr and sname.lower().startswith('chr'):
					sname = sname[3:]
				if self.restrictToSequences != None and not sname in self.restrictToSequences:
					continue
				fIn.seek(seqInfo['offset'])
				nnt, nblk = struct.unpack('II', fIn.read(8))
				blkStarts = list(struct.unpack(''.join('I' for _ in range(nblk)), fIn.read(4 * nblk)))
				blkSizes = list(struct.unpack(''.join('I' for _ in range(nblk)), fIn.read(4 * nblk)))
				nmask = struct.unpack('I', fIn.read(4))[0]
				maskStarts = list(struct.unpack(''.join('I' for _ in range(nmask)), fIn.read(4 * nmask)))
				maskSizes = list(struct.unpack(''.join('I' for _ in range(nmask)), fIn.read(4 * nmask)))
				reserved = struct.unpack('I', fIn.read(4))[0]
				Nblocks = sorted([ stream2bitNBlock(bstart, bsize) for bstart, bsize in zip(blkStarts, blkSizes) ], key = lambda x: x.start)
				yield stream2bitSeq(sname, fIn, self.wantBlockSize, Nblocks, nnt)
		return None
	
	def __str__(self):
		return '2bit sequence stream<%s>'%(self.name)
	
	def __repr__(self):
		return self.__str__()

# Streams a 2bit file in blocks.
cpdef stream2bit(path, wantBlockSize = 5000, spacePrune = True, dropChr = True, restrictToSequences = None):
	"""
	Streams a 2bit file.
	
	:param path: Path to the input file.
	:param wantBlockSize: Desired block size.
	:param spacePrune: Whether or not to prune sequence name spaces.
	:param dropChr: Whether or not to prune sequence name "chr"-prefixes.
	:param restrictToSequences: List of sequence names to restrict to/focus on.
	
	:type path: str
	:type wantBlockSize: int, optional
	:type spacePrune: bool, optional
	:type dropChr: bool, optional
	:type restrictToSequences: list, optional
	
	:return: Generated sequence stream.
	:rtype: sequenceStream
	"""
	return sequenceStream2bit(path, wantBlockSize, spacePrune, dropChr, restrictToSequences)

# Gets a sequence stream based on a path, deciding on the format from the path.
cpdef getSequenceStreamFromPath(path, wantBlockSize = 5000, spacePrune = True, dropChr = True, restrictToSequences = None):
	"""
	Creates a sequence stream from a path, deducing the file type.
	
	:param path: Path to the input file.
	:param wantBlockSize: Desired block size.
	:param spacePrune: Whether or not to prune sequence name spaces.
	:param dropChr: Whether or not to prune sequence name "chr"-prefixes.
	:param restrictToSequences: List of sequence names to restrict to/focus on.
	
	:type path: str
	:type wantBlockSize: int, optional
	:type spacePrune: bool, optional
	:type dropChr: bool, optional
	:type restrictToSequences: list, optional
	
	:return: Generated sequence stream.
	:rtype: sequenceStream
	"""
	l = path.lower()
	if l.endswith(".fa") or l.endswith(".fasta"):
		return streamFASTA(path, wantBlockSize = wantBlockSize, spacePrune = spacePrune, dropChr = dropChr , restrictToSequences = restrictToSequences)
	elif l.endswith(".fa.gz") or l.endswith(".fasta.gz"):
		return streamFASTAGZ(path, wantBlockSize = wantBlockSize, spacePrune = spacePrune, dropChr = dropChr , restrictToSequences = restrictToSequences)
	elif l.endswith(".2bit") or l.endswith(".2b"):
		return stream2bit(path, wantBlockSize = wantBlockSize, spacePrune = spacePrune, dropChr = dropChr , restrictToSequences = restrictToSequences)
	else:
		raise Exception('Unrecognized sequence format')

# Streams sequence windows from a sequence set or sequence stream, with coordinates and window sequence.
def streamSequenceWindows(src, int windowSize, int windowStep):
	"""
	Yields sequence windows for a sequence set, sequence stream or a path to a sequence file.
	
	:param src: Input sequences.
	:param windowSize: Window size.
	:param windowStep: Window step size.
	
	:type src: sequences/sequenceStream/str
	:type windowSize: int
	:type windowStep: int
	
	:return: Sequence windows.
	:rtype: sequence (yield)
	"""
	cdef str cbuf
	cdef int cstart
	cdef object cseq
	cdef str blk
	if isinstance(src, str):
		src = getSequenceStreamFromPath(src)
	if isinstance(src, sequences):
		for cseq in src:
			cbuf = '' + cseq.seq
			cstart = 0
			while len(cbuf) >= windowSize:
				yield sequence(None, cbuf[:windowSize], sourceRegion = region(cseq.name, cstart, cstart+windowSize-1))
				cbuf = cbuf[windowStep:]
				cstart += windowStep
	elif isinstance(src, sequenceStream):
		for cseq in src:
			cbuf = ''
			cstart = 0
			for blk in cseq:
				cbuf += blk
				while len(cbuf) >= windowSize:
					yield sequence(None, cbuf[:windowSize], sourceRegion = region(cseq.name, cstart, cstart+windowSize-1))
					cbuf = cbuf[windowStep:]
					cstart += windowStep

# Returns sliding window coordinates for a sequence set or sequence stream.
def getSequenceWindowRegions(object src, int windowSize, int windowStep):
	"""
	Generates sequence window regions for a sequence set or stream.
	
	:param stream: Input sequences.
	:param windowSize: Window size.
	:param windowStep: Window step size.
	
	:type src: sequences/sequenceStream
	:type windowSize: int
	:type windowStep: int
	
	:return: Sequence window regions.
	:rtype: regions
	"""
	cdef dict seqLengths
	cdef list rset
	cdef str cseq
	cdef int cstart
	if isinstance(src, str):
		src = getSequenceStreamFromPath(src)
	seqLengths = src.sequenceLengths()
	rset = []
	for cseq in sorted(seqLengths.keys()):
		cstart = 0
		while cstart + windowSize <= seqLengths[cseq]:
			rset.append( region(cseq, cstart, cstart+windowSize-1) )
			cstart += windowStep
	return regions('Windows (Window size: %d bp; step size: %d bp)'%(windowSize, windowStep), rset, initialSort = False)


###########################################################################
# Sequence generation

# Represents individual sequences within a generator stream
class sequenceStreamGeneratorSequence:
	
	def __init__(self, model, wantBlockSize, wantSize, number, seed):
		self.model = model
		self.wantSize = wantSize
		self.wantBlockSize = wantBlockSize
		self.name = 'Random sequence %d (seed = %d)'%(number, seed)
	
	def __iter__(self):
		return self.stream()
	
	def __str__(self):
		return 'Generator sequence stream sequence<%s, %d>'%(str(self.model), self.wantSize)
	
	def __repr__(self):
		return self.__str__()
	
	def stream(self):
		while self.wantSize > 0:
			ng = min(self.wantBlockSize, self.wantSize)
			self.wantSize -= ng
			yield self.model.generate(ng).seq

# Represents stream of 2bit sequences.
class sequenceStreamGenerator(sequenceStream):
	
	def __init__(self, model, wantBlockSize, wantN, wantSize):
		self.model = model
		self.wantN = wantN
		self.wantSize = wantSize
		self.wantBlockSize = wantBlockSize
		self.name = str(model)
	
	def __iter__(self):
		seed = random.randint(0, 0xFFFFFFFF)
		random.seed(seed)
		for i in range(self.wantN):
			yield sequenceStreamGeneratorSequence(model = self.model, wantBlockSize = self.wantBlockSize, wantSize = self.wantSize, number = (i+1), seed = seed)
	
	def __str__(self):
		return 'Generator sequence stream<%s>'%str(self.model)
	
	def __repr__(self):
		return self.__str__()

# Returns a weighted random choice from a list of tuples of choices and weights.
def weightedRandomChoice(choices):
	r = random.random()
	cw = 0.0
	for c, w in choices:
		cw += w
		if cw >= r:
			return c
	raise Exception('Choice weights sum up to more than 1.0')

# Represents sequence streams
cdef class sequenceGenerator:
	"""
	The `sequenceGenerator` class is an abstract class for sequence generators.
	"""
	
	def __init__(self):
		pass
	
	# Generates a set of sequences
	def generateSet(self, n, length):
		"""
		:param n: Number of sequences to generate.
		:param length: Length of each sequence to generate.
		:type n: int
		:type length: int
		
		:return: Sequence set of `n` randomly generated sequences, each of length `length`.
		:rtype: sequences
		"""
		seed = random.randint(0, 0xFFFFFFFF)
		random.seed(seed)
		return sequences('Generated set <Sequences: %d; Length each: %d; Model: %s; Seed: %d>'%(n, length, str(self), seed), [ self.generate(length) for _ in range(n) ])
	
	# Generates a stream of sequences
	def generateStream(self, n, length, wantBlockSize = 5000):
		"""
		:param n: Number of sequences to generate.
		:param length: Length of each sequence to generate.
		:param wantBlockSize: Desired block size.
		:type n: int
		:type length: int
		:type wantBlockSize: int, optional
		
		:return: Sequence stream of `n` randomly generated sequences, each of length `length`.
		:rtype: sequencesStream
		"""
		seed = random.randint(0, 0xFFFFFFFF)
		random.seed(seed)
		return sequenceStreamGenerator(model = self, wantBlockSize = wantBlockSize, wantN = n, wantSize = length)
	
	# Generates sequences to a FASTA-file
	def generateFASTA(self, str path, int n, int length):
		""" Generates sequences and saves them to a FASTA file.
		
		:param path: Output path.
		:param n: Number of sequences to generate.
		:param length: Length of each sequence to generate.
		:type path: str
		:type n: int
		:type length: int
		"""
		cdef int i
		cdef sequence seq
		seed = random.randint(0, 0xFFFFFFFF)
		random.seed(seed)
		with open(path, 'w') as fOut:
			#for seq in self.sequences:
			for i in range(n):
				seq = self.generate(length)
				fOut.write(">%s\n%s\n"%(seq.name.replace(' ', '_'), re.sub("(.{65})","\\1\n", seq.seq, 0, re.DOTALL)))

# Markov chain sequence model.
# Optimized using spectrum and lookup table. Tested for functional equivalence with original Markov chain code.
cdef class MarkovChain(sequenceGenerator):
	"""
	The `MarkovChain` class trains a Markov chain for generating sequences.
	
	:param trainingSequences: Training sequences.
	:param degree: Markov chain degree. Default is 4.
	:param pseudoCounts: Whether or not to use pseudocounts. Default is `True`.
	:param addReverseComplements: Whether or not to add reverse complements. Default is `True`.
	
	:type trainingSequences: sequences/sequenceStream/str
	:type degree: int, optional
	:type pseudoCounts: bool, optional
	:type addReverseComplements: bool, optional
	"""
	
	def __init__(self, trainingSequences, degree = 4, pseudoCounts = True, addReverseComplements = True):
		self.degree = degree
		self.pseudoCounts = pseudoCounts
		self.addReverseComplements = addReverseComplements
		self.trainingSequences = trainingSequences
		self.nGenerated = 0
		self.prepared = False
		# Initialize spectrum
		self.spectrum = [ 0 for _ in range( 4 ** (degree+1) ) ]
		# Train
		if trainingSequences == None:
			return
		if isinstance(trainingSequences, str):
			trainingSequences = getSequenceStreamFromPath(trainingSequences)
		if type(trainingSequences).__name__ == 'genome':
			trainingSequences = trainingSequences.sequences
		if isinstance(trainingSequences, sequences) or isinstance(trainingSequences, list):
			self._trainOnSequences(trainingSequences)
			self._prepare()
		elif isinstance(trainingSequences, sequenceStream):
			self._trainOnSequenceStream(trainingSequences)
			self._prepare()
	
	def __str__(self):
		return 'Markov Chain<Degree: %d; Pseudocounts: %d; Add reverse complements: %s; Training set: %s>'%(self.degree, self.pseudoCounts, 'Yes' if self.addReverseComplements else 'No', self.trainingSequences)
	
	def __repr__(self):
		return self.__str__()
	
	# Prepares model for use
	def _prepare(self):
		"""
		Prepares model for use.
		"""
		#cdef int nspectrum, nOccT, ki, kiRC, tki
		cdef int nspectrum, ki, kiRC, tki
		# Add reverse complements
		if self.addReverseComplements:
			newspectrum = [ 0 for _ in range(len(self.spectrum)) ]
			nspectrum = self.degree + 1
			for ki in range(len(self.spectrum)):
				kiRC = 0
				tki = ki
				for x in range(nspectrum):
					kiRC = (kiRC << 2) | ((tki & 3)^1)
					tki = tki >> 2
				newspectrum[ki] = self.spectrum[ki] + self.spectrum[kiRC]
			self.spectrum = newspectrum
		# Add pseudocounts
		if self.pseudoCounts > 0:
			for i in range(len(self.spectrum)):
				self.spectrum[i] += self.pseudoCounts
		# Generate normalized sequence spectrum with occurrence probabilities
		nOccT = sum( self.spectrum )
		self.initialDistribution = []
		self.probspectrum = [ [] for _ in range( 4 ** self.degree ) ]
		for ki in range(len(self.probspectrum)):
			kis = ki << 2
			obsT = sum( self.spectrum[ kis | nti ] for nti in range(4) )
			if obsT > 0:
				self.probspectrum[ki] = [ (nti, self.spectrum[ kis | nti ] / obsT) for nti in range(4) ]
				self.initialDistribution.append( (ki, obsT / nOccT) )
			else:
				self.probspectrum[ki] = [ (nti, 0.0) for nti in range(4) ]
		# Construct comparable spectrum for debugging purposes
		self.comparableSpectrum = {}
		for ki in range(len(self.probspectrum)):
			kmer = ''.join( kSpectrumIndex2NT[(ki >> ((self.degree - 1 - x)*2)) & 3] for x in range(self.degree) )
			
			self.comparableSpectrum[kmer] =  { nt: self.spectrum[(ki << 2) | kSpectrumNT2Index[nt]] for nt in [ 'A', 'T', 'G', 'C' ] }
		self.prepared = True
	
	# Trains model on a sequence set
	def _trainOnSequences(self, trainingSequences):
		"""Trains on sequences.
		
		:param trainingSequences: Training sequences.
		:type trainingSequences: sequences
		"""
		if self.prepared:
			raise Exception('Tried to train Markov model that is already prepared for use')
		cdef list spectrum
		cdef int nspectrum, bitmask, ki, nnt
		cdef object seq
		cdef str seqseq
		cdef str c
		cdef int v
		cdef list convtable
		spectrum = self.spectrum
		nspectrum = self.degree + 1
		bitmask = (4 ** (self.degree + 1))-1
		convtable = [ -1 for _ in range(256) ]
		for k in kSpectrumNT2Index.keys():
			convtable[<int>(ord(k))] = <int>kSpectrumNT2Index[k]
		for seq in trainingSequences:
			ki = 0
			nnt = 0
			seqseq = seq.seq
			for c in seqseq:
				v = convtable[<int>ord(c)]
				if v == -1:
					ki = 0
					nnt = 0
					continue
				ki = ( ( ki << 2 ) | v ) & bitmask
				nnt += 1
				if nnt >= nspectrum:
					spectrum[ki] += 1
					nnt -= 1
	
	# Trains model on a sequence stream
	def _trainOnSequenceStream(self, trainingSequences):
		"""Trains on sequence stream.
		
		:param trainingSequences: Training sequences.
		:type trainingSequences: sequenceStream
		"""
		if self.prepared:
			raise Exception('Tried to train Markov model that is already prepared for use')
		cdef list spectrum
		cdef int nspectrum, bitmask, ki, nnt
		cdef object seq
		cdef str blk
		cdef str c
		cdef int v
		cdef list convtable
		spectrum = self.spectrum
		nspectrum = self.degree + 1
		bitmask = (4 ** (self.degree + 1))-1
		convtable = [ -1 for _ in range(256) ]
		for k in kSpectrumNT2Index.keys():
			convtable[<int>(ord(k))] = <int>kSpectrumNT2Index[k]
		for seq in trainingSequences:
			ki = 0
			nnt = 0
			for blk in seq:
				for c in blk:
					v = convtable[<int>ord(c)]
					if v == -1:
						ki = 0
						nnt = 0
						continue
					ki = ( ( ki << 2 ) + v ) & bitmask
					nnt += 1
					if nnt >= nspectrum:
						spectrum[ki] += 1
						nnt -= 1
	
	# Generates a sequence
	def generate(self, int length):
		""" Generates a sequence.
		
		:param length: Length of the sequence to generate.
		:type length: int
		
		:return: Random sequence.
		:rtype: sequence
		"""
		if not self.prepared:
			self.prepare()
		cdef int state, andMask, nnt, degree
		cdef str seq, kmer
		#
		cdef float r, cw, w
		cdef list choices
		cdef int c
		#
		degree = self.degree
		seed = random.randint(0, 0xFFFFFFFF)
		random.seed(seed)
		state = weightedRandomChoice(self.initialDistribution)
		kmer = ''.join( kSpectrumIndex2NT[(state >> ((self.degree - 1 - x)*2)) & 3] for x in range(self.degree) )
		seq = '' + kmer
		andMask = ((4 ** (degree))-1)
		
		##################################
		cdef char* gbuf = <char*> malloc(length+1)
		cdef char* index2NT = <char*> malloc(4)
		index2NT[0] = b'A'
		index2NT[1] = b'T'
		index2NT[2] = b'G'
		index2NT[3] = b'C'
		cdef char* cseq = gbuf
		for c in range(degree):
			cseq[0] = ord(seq[c])
			cseq += 1
		for _ in range(length - degree):
			r = random.random()
			cw = 0.0
			choices = self.probspectrum[state]
			for c, w in choices:
				cw += w
				if cw >= r:
					nnt = c
					break
			state = ((state << 2) | nnt) & andMask
			cseq[0] = index2NT[nnt]
			cseq += 1
		cseq[0] = 0
		seq = (<bytes>gbuf).decode('UTF-8')
		free(gbuf)
		free(index2NT)
		self.nGenerated += 1
		return sequence('Generated sequence <Length: %d; Model: %s; Seed: %d>'%(length, str(self), seed), seq)

# IID sequence model
cdef class IID(sequenceGenerator):
	"""
	The `IID` class trains an i.i.d. model for generating sequences.
	
	:param trainingSequences: Training sequences.
	:param pseudoCounts: Whether or not to use pseudocounts. Default is `True`.
	:param addComplements: Whether or not to add complements. Default is `True`.
	
	:type trainingSequences: sequences/sequenceStream/str
	:type pseudoCounts: bool, optional
	:type addComplements: bool, optional
	"""
	
	def __init__(self, trainingSequences, pseudoCounts = True, addComplements = True):
		self.pseudoCounts = pseudoCounts
		self.addComplements = addComplements
		self.trainingSequences = trainingSequences
		self.nGenerated = 0
		self.prepared = False
		# Initialize spectrum
		self.spectrum = { nt: 0 for nt in nucleotides }
		# Train
		if trainingSequences == None:
			return
		if isinstance(trainingSequences, str):
			trainingSequences = getSequenceStreamFromPath(trainingSequences)
		if type(trainingSequences).__name__ == 'genome':
			trainingSequences = trainingSequences.sequences
		if isinstance(trainingSequences, sequences) or isinstance(trainingSequences, list):
			self._trainOnSequences(trainingSequences)
			self._prepare()
		elif isinstance(trainingSequences, sequenceStream):
			self._trainOnSequenceStream(trainingSequences)
			self._prepare()
	
	def __str__(self):
		return 'IID<Pseudocounts: %d; Add complements: %s; Training set: %s>'%(self.pseudoCounts, 'Yes' if self.addComplements else 'No', self.trainingSequences)
	
	def __repr__(self):
		return self.__str__()
	
	# Trains model on a sequence seq
	def _trainOnSequences(self, trainingSequences):
		"""Trains on sequences.
		
		:param trainingSequences: Training sequences.
		:type trainingSequences: sequences
		"""
		cdef object seq
		cdef str nt
		if self.prepared:
			raise Exception('Tried to train IID model that is already prepared for use')
		for seq in trainingSequences:
			for nt in seq.seq:
				if not nt in nucleotides: continue
				self.spectrum[nt] += 1
	
	# Trains model on a sequence stream
	def _trainOnSequenceStream(self, trainingSequences):
		"""Trains on sequence stream.
		
		:param trainingSequences: Training sequences.
		:type trainingSequences: sequenceStream
		"""
		cdef object seq
		cdef str blk, nt
		if self.prepared:
			raise Exception('Tried to train IID model that is already prepared for use')
		for seq in trainingSequences:
			for blk in seq:
				for nt in blk:
					if not nt in nucleotides: continue
					self.spectrum[nt] += 1
	
	# Prepares model for use
	def _prepare(self):
		"""
		Prepares model for use.
		"""
		# Add complements
		if self.addComplements:
			newSpectrum = { nt: (self.spectrum[nt] + self.spectrum[complementaryNucleotides[nt]]) for nt in nucleotides }
			self.spectrum = newSpectrum
		# Add pseudocounts
		if self.pseudoCounts > 0:
			for nt in ['A','T','G','C']:
				self.spectrum[nt] += self.pseudoCounts
		# Generate normalized sequence spectrum with occurrence probabilities
		ntT = sum( self.spectrum[nt] for nt in nucleotides )
		self.ntDistribution = [ (nt, float(self.spectrum[nt])/ ntT) for nt in nucleotides ]
		self.prepared = True
	
	# Generates a sequence
	def generate(self, length):
		""" Generates a sequence.
		
		:param length: Length of the sequence to generate.
		:type length: int
		
		:return: Random sequence.
		:rtype: sequence
		"""
		if not self.prepared:
			self.prepare()
		seed = random.randint(0, 0xFFFFFFFF)
		random.seed(seed)
		seq = ''.join( weightedRandomChoice(self.ntDistribution) for _ in range(length) )
		self.nGenerated += 1
		return sequence('Generated sequence <Length: %d, Model: %s; Seed: %d>'%(length, str(self), seed), seq)


