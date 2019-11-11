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
from .common import nucleotides, complementaryNucleotides, getReverseComplementaryDNASequence, IUPACNucleotideCodes, IUPACNucleotideCodeSemantics


############################################################################
# Motifs

# Represents a DNA sequence motif occurrence.
cdef class motifOccurrence:
	"""
	Represents motif occurrences.
	
	:param motif: The motif.
	:param seq: Name of the sequence the motif occurred on.
	:param start: Occurrence start nucleotide.
	:param end: Occurrence end nucleotide.
	:param strand: Occurrence strand. `True` is the +/forward strand, and `False` is the -/backward strand.
	
	:type motif: object
	:param seq: str
	:param start: int
	:param end: int
	:param strand: bool
	
	After constructing a `biomarkers` object, for a set of biomarkers `BM`, `len(BM)` gives the number of markers, `BM['FACTOR_NAME']` gives the merged set of regions for factor `FACTOR_NAME`, and `[ x for x in BM ]` gives the list of merged regions per biomarker contained in `BM`.
	"""
	
	__slots__ = 'motif', 'seq', 'start', 'end', 'strand'
	
	def __init__(self, motif, seq, start, end, strand):
		self.motif, self.seq, self.start, self.end, self.strand = motif, seq, start, end, strand
	
	def __str__(self):
		return 'Motif occurrence<Motif: %s; Coordinates: %s:%d..%d (%s)>'%(self.motif.name, self.seq.name, self.start, self.end, '+' if self.strand else '-')
	
	def __repr__(self):
		return self.__str__()

# Represents motif sets.
class motifs:
	"""
	The `motifs` class represents a set of motifs.
	
	:param name: Name of the motif set.
	:param motifs: List of motifs.
	
	:type name: str
	:param motifs: list
	"""
	
	__slots__ = 'name', 'motifs'
	
	def __init__(self, name, m):
		self.name = name
		self.motifs = m
	
	def __iter__(self):
		return self.motifs.__iter__()
	
	def __getitem__(self, i):
		return self.motifs[i]
	
	def __len__(self):
		return len(self.motifs)
	
	def __str__(self):
		return 'Motif set<%s>'%(self.name)
	
	def __repr__(self):
		return self.__str__()
	
	def __add__(self, other):
		return motifs('%s + %s'%(self.name, other.name), self.motifs + other.motifs)
	
	@staticmethod
	def getRingrose2003Motifs_GTGT():
		"""
		Preset for generating the Ringrose et al. (2003) motif set, with GTGT added (as in Bredesen et al. 2019).
		"""
		return motifs('Ringrose et al. 2003 + GTGT', [
				IUPACMotif('En', 'GSNMACGCCCC', 1),
				IUPACMotif('G10', 'GAGAGAGAGA', 1),
				IUPACMotif('GAF', 'GAGAG', 0),
				IUPACMotif('PF', 'GCCATHWY', 0),
				IUPACMotif('PM', 'CNGCCATNDNND', 0),
				IUPACMotif('PS', 'GCCAT', 0),
				IUPACMotif('Z', 'YGAGYG', 0),
				IUPACMotif('GTGT', 'GTGT', 0)
			])
	
	@staticmethod
	def getRingrose2003Motifs():
		"""
		Preset for generating the Ringrose et al. (2003) motif set.
		"""
		return motifs('Ringrose et al. 2003 + GTGT', [
				IUPACMotif('En', 'GSNMACGCCCC', 1),
				IUPACMotif('G10', 'GAGAGAGAGA', 1),
				IUPACMotif('GAF', 'GAGAG', 0),
				IUPACMotif('PF', 'GCCATHWY', 0),
				IUPACMotif('PM', 'CNGCCATNDNND', 0),
				IUPACMotif('PS', 'GCCAT', 0),
				IUPACMotif('Z', 'YGAGYG', 0)
			])



############################################################################
# IUPAC motifs

# Represents an IUPAC nucleotide code DNA sequence motif.
cdef class IUPACMotif:
	"""
	Represents an IUPAC motif.
	
	:param name: Name of the motif.
	:param motif: Motif sequence, in IUPAC nucleotide codes (https://www.bioinformatics.org/sms/iupac.html).
	:type nmismatches: The number of mismatches allowed.
	
	:type name: str
	:type motif: str
	:type nmismatches: int
	"""
	
	__slots__ = 'name', 'motif', 'nmismatches', 'regexMotif', 'regexMotifRC', 'c', 'cRC'
	
	def __init__(self, name, motif, nmismatches = 0):
		self.name, self.motif, self.nmismatches = name, motif, nmismatches
		s = self.motif
		if self.nmismatches > 1:
			raise Exception('More than one mismatch per motif is currently not supported')
		for c in motif:
			if not c in nucleotides + IUPACNucleotideCodes:
				raise Exception('Invalid nucleotide in IUPAC motif "' + motif + '"')
		if self.nmismatches == 1:
			s = '|'.join([ motif[:i] + "N" + motif[i+1:] for i in range(len(motif)) ])
		for k in IUPACNucleotideCodeSemantics.keys():
			s = re.sub(k, '('+'|'.join(IUPACNucleotideCodeSemantics[k])+')', s)
		self.regexMotif = s
		self.regexMotifRC = getReverseComplementaryDNASequence(s)
		self.c = re.compile(self.regexMotif)
		self.cRC = re.compile(self.regexMotifRC)
		self.cachedSequence = None
		self.cachedOcc = []
	
	def __str__(self):
		return 'IUPACMotif<%s (%s, %d mismatches allowed)>'%(self.name, self.motif, self.nmismatches)
	
	def __repr__(self): return self.__str__()
	
	# Finds occurrences of the motif in a given sequence.
	def findOccurrences(self, object seq, bool cache=True):
		"""
		Finds the occurrences of the motif in a sequence.
		
		:param seq: The sequence.
		:param cache: Whether or not to cache.
		:type seq: sequence
		:type cache: bool
		
		:return: List of motif occurrences
		:rtype: list
		"""
		# Sorting-independent regular expression motif occurrence search
		cdef int posF, posRC
		cdef object resultF, resultRC
		cdef str seqstr
		cdef list ret
		if cache:
			if self.cachedSequence == seq:
				return self.cachedOcc
		ret = []
		posF, posRC = 0, 0
		resultF, resultRC = None, None
		seqstr = seq.seq
		while True:
			# Search for next motif occurrence in a merge-sorted manner
			if resultF == None:
				resultF = self.c.search(seqstr, posF)
				if resultF != None:
					posF = resultF.start() + 1
			if resultRC == None:
				resultRC = self.cRC.search(seqstr, posRC)
				if resultRC != None:
					posRC = resultRC.start() + 1
			# If no occurrences are found on either strand, then break
			if resultF == None and resultRC == None:
				break
			# If found on both strands, add the first
			if resultF != None and resultRC != None:
				if posF < posRC:
					ret.append( motifOccurrence(self, seq, resultF.start(), resultF.end(), True) )
					resultF = None
				else:
					ret.append( motifOccurrence(self, seq, resultRC.start(), resultRC.end(), False) )
					resultRC = None
			# If found only on the forward or the reverse complement, add those
			elif resultF != None:
				ret.append( motifOccurrence(self, seq, resultF.start(), resultF.end(), True) )
				resultF = None
			elif resultRC != None:
				ret.append(  motifOccurrence(self, seq, resultRC.start(), resultRC.end(), False) )
				resultRC = None
		if cache:
			self.cachedSequence = seq
			self.cachedOcc = ret
		return ret


############################################################################
# Position Weight Matrix motifs

cdef class PWMMotif:
	"""
	Represents an Position Weight Matrix (PWM) motif.
	
	:param name: Name of the motif.
	:param pwm: Position Weight Matrix.
	:param path: Path to file that the PWM was loaded from.
	:type threshold: Threshold for making occurrence predictions.
	
	:type name: str
	:type pwm: str
	:type path: str
	:type threshold: float
	"""
	
	def __init__(self, name, pwm, path = '', threshold = 0.0):
		self.name, self.path, self.threshold = name, path, threshold
		self.bPWMF = NULL
		self.bPWMRC = NULL
		self.maxScoreLeftF = NULL
		self.maxScoreLeftRC = NULL
		self.setPWM(pwm)
		self.cachedSequence = None
		self.cachedOcc = []
	
	def __dealloc__(self):
		if self.bPWMF:
			free(self.bPWMF)
			self.bPWMF = NULL
		if self.bPWMRC:
			free(self.bPWMRC)
			self.bPWMRC = NULL
		if self.maxScoreLeftF:
			free(self.maxScoreLeftF)
			self.maxScoreLeftF = NULL
		if self.maxScoreLeftRC:
			free(self.maxScoreLeftRC)
			self.maxScoreLeftRC = NULL
	
	def __str__(self):
		return 'PWMMotif<%s (from %s)>'%(self.name, self.path)
	
	def __repr__(self): return self.__str__()
	
	def getEValueThreshold(self, bgModel, Evalue = 1.0, EvalueUnit = 1000.0, seedThreshold = 0.0, iterations = 100):
		""" Calibrates the threshold for a desired E-value.
		
		:param bgModel: Background model to use for calibration.
		:param Evalue: The desired E-value.
		:param EvalueUnit: E-value unit.
		:param seedThreshold: Seed threshold.
		:param iterations: Number of iterations.
		
		:type bgModel: object
		:type Evalue: float
		:type EvalueUnit: float
		:type seedThreshold: float
		:type iterations: int
		
		:return: Threshold
		:rtype: float
		"""
		self.threshold = seedThreshold
		thresholds = []
		for _ in range(iterations):
			seq = bgModel.generate( int(EvalueUnit / Evalue) )
			occ = self.findOccurrences(seq, cache=False)
			maxScore = max( o.score for o in occ )
			thresholds.append( maxScore - 0.0000000000000000000000000000000001 )
		return sum(thresholds) / len(thresholds)
	
	def setPWM(self, pwm):
		""" Sets the Position Weight Matrix.
		
		:param pwm: Position Weight Matrix.
		
		:type pwm: list
		"""
		self.PWMF = pwm
		self.PWMRC = [ { nt: x[complementaryNucleotides[nt]] for nt in nucleotides } for x in reversed(self.PWMF) ]
		nNT = len(pwm)
		if self.bPWMF:
			free(self.bPWMF)
			self.bPWMF = NULL
		if self.bPWMRC:
			free(self.bPWMRC)
			self.bPWMRC = NULL
		if self.maxScoreLeftF:
			free(self.maxScoreLeftF)
			self.maxScoreLeftF = NULL
		if self.maxScoreLeftRC:
			free(self.maxScoreLeftRC)
			self.maxScoreLeftRC = NULL
		cdef double* bPWMF = <double*> malloc(sizeof(double) * 4 * nNT)
		cdef double* bPWMRC = <double*> malloc(sizeof(double) * 4 * nNT)
		self.maxScoreLeftF = <double*> malloc(sizeof(double) * nNT)
		self.maxScoreLeftRC = <double*> malloc(sizeof(double) * nNT)
		for i in range(nNT):
			for c,nt in enumerate(nucleotides):
				bPWMF[ (i<<2) + c ] = self.PWMF[i][nt]
				bPWMRC[ (i<<2) + c ] = self.PWMRC[i][nt]
			# We calculate and store the maximum remaining score for each position, to be able to terminate scoring prematurely when the score never will reach the set threshold
			self.maxScoreLeftF[i] = sum( max( self.PWMF[x][nt] for nt in nucleotides ) for x in range(i+1, nNT) )
			self.maxScoreLeftRC[i] = sum( max( self.PWMRC[x][nt] for nt in nucleotides ) for x in range(i+1, nNT) )
		self.bPWMF = bPWMF
		self.bPWMRC = bPWMRC
	
	def findOccurrences(self, sequence seq, bool cache=True):
		"""
		Finds the occurrences of the motif in a sequence.
		
		:param seq: The sequence.
		:param cache: Whether or not to cache.
		:type seq: sequence
		:type cache: bool
		
		:return: List of motif occurrences
		:rtype: list
		"""
		if cache:
			if self.cachedSequence == seq:
				return self.cachedOcc
		cdef double scoreF, scoreRC, threshold
		cdef list ret
		cdef int nNT, i, mPos, posInd, slen
		cdef bytes bseq
		cdef unsigned char bnt
		threshold = self.threshold
		ret = []
		nNT = len(self.PWMF)
		slen = len(seq)
		bseq = seq.getBytesIndexed()
		cdef double* bPWMF = self.bPWMF
		cdef double* bPWMRC = self.bPWMRC
		cdef double* maxScoreLeftF = self.maxScoreLeftF
		cdef double* maxScoreLeftRC = self.maxScoreLeftRC
		for i in range(slen - nNT + 1):
			scoreF = 0.0
			scoreRC = 0.0
			for mPos in range( nNT ):
				bnt = bseq[ i + mPos ]
				if bnt == 0xFF: break
				posInd = (mPos << 2) + bnt
				scoreF += bPWMF[posInd]
				scoreRC += bPWMRC[posInd]
				# If the sum of the current score and the maximum last remaining score is below the threshold, we can stop scoring, as there is no way for the score to reach the threshold
				if scoreF + maxScoreLeftF[mPos] <= threshold and scoreRC + maxScoreLeftRC[mPos] <= threshold:
					break
			if scoreF > threshold:
				ret.append( motifOccurrence(self, seq, i, i+nNT, True, scoreF) )
			if scoreRC > threshold:
				ret.append( motifOccurrence(self, seq, i, i+nNT, False, scoreRC) )
		if cache:
			self.cachedSequence = seq
			self.cachedOcc = ret
		return ret

def loadMEMEPWMDatabase(path):
	MEMEAlphabet = None
	rmotifs = []
	with open(path, 'r') as fIn:
		if not next(fIn).strip() in ['MEME version 4', 'MEME version 4.5.0']:
			raise Exception('Unsupported MEME motif database version')
		for l in fIn:
			if l.startswith('ALPHABET= '):
				MEMEAlphabet = [ nt for nt in l[10:].strip() ]
			if l.startswith('strands: '):
				continue
			if l.startswith('MOTIF '):
				names = l[6:].strip().split(' ')
				rmotifs.append( PWMMotif( '/'.join(names) if len(names) > 1 else names[0], [], path.split('/')[-1] ) )
			if l.startswith('letter-probability matrix: '):
				if len(rmotifs) == 0:
					raise Exception('MEME motif database syntax error')
				remainder = l[27:].strip()
				params = {}
				while len(remainder) > 0:
					try:
						i = remainder.index('= ')
						sname = remainder[:i]
						b = remainder[i+2:]
						i = b.index(' ')
						svalue = b[:i]
						remainder = b[i+1:]
						params[sname] = svalue
					except ValueError:
						break
				if params['alength'] != '4':
					raise Exception('Invalid MEME nucleotide motif')
				weights = []
				for pos in range(int(params['w'])):
					row = { nt: float(e.strip()) for nt, e in zip(MEMEAlphabet, next(fIn).strip().split('\t')) }
					weights.append(row)
				#rmotifs[-1].PWM = weights
				rmotifs[-1].setPWM(weights)
			if l == '': continue
	return motifs('MEME database: ' + path.split('/')[-1], rmotifs)

