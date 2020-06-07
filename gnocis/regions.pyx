# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from __future__ import division
import random
import gzip
from libcpp cimport bool
from .sequences cimport *
from .ioputil import nctable


############################################################################
# Regions

# Represents regions.
cdef class region:
	"""
	The `region` class represents a region within a sequence. A region has a sequence name, a start and end, and a strandedness, with optional additional annotation.
	
	:param seq: Sequence name.
	:param start: Start coordinate.
	:param end: End coordinate (inclusive).
	:param strand: Strandedness. `True` for the + (forward) strand (default), and `False` for the - (reverse) strand.
	:param score: Score.
	:param source: Soure.
	:param feature: Feature.
	:param group: Group.
	:param dropChr: `True` if a leading `chr` in `seq` is to be dropped (default).
	
	:type seq: str
	:type start: int
	:type end: int
	:type strand: bool, optional
	:type score: float, optional
	:type source: str, optional
	:type feature: str, optional
	:type group: str, optional
	:type dropChr: bool, optional
	
	The length of a region `r` can be calculated with `len(r)`.
	"""
	
	__slots__ = ('seq', 'start', 'end', 'strand', 'score', 'source', 'feature', 'markers', 'group', 'ext')
	
	def __init__(self, seq, start, end, strand = True, score = 0.0, source = '.', feature = '.', group = '.', dropChr = True):
		if dropChr:
			if seq.lower().startswith('chr'):
				seq = seq[3:]
		self.seq, self.start, self.end, self.strand, self.score = seq, start, end, strand, score
		if start > end: self.start, self.end = end, start
		self.source, self.feature, self.group = source, feature, group
		self.ext = None
	
	def singleton(self):
		return regions(str(self), [ self ])
	
	def center(self):
		return int((self.start + self.end) / 2)
	
	def __len__(self):
		return self.end+1-self.start
	
	def __str__(self):
		rname = '%s:%d..%d (%s)'%(self.seq, self.start, self.end, '+' if self.strand else '-')
		ext = [ 'Score: %f'%self.score ]
		if self.source != '.':
			ext.append( 'Source: %s'%self.source )
		if self.feature != '.':
			ext.append( 'Feature: %s'%self.feature )
		return 'Region<%s (%s)>'%(rname, '; '.join(ext))
	
	def __repr__(self):
		return self.__str__()
	
	def bstr(self):
		"""
		:return: Returns the region formatted as a string: `seq:start..end (strand)`.
		:rtype: str
		"""
		return '%s:%d..%d (%s)'%(self.seq, self.start, self.end, '+' if self.strand else '-')

# Represents region sets. Supports operands '+' for adding regions from two sets with no merging, '|' for merging two sets, and '^' for excluding the regions in the latter set from the first.
cdef class regions:
	"""
	The `regions` class represents a set of regions.
	
	:param name: Name for the sequence set.
	:param rgns: List of regions to include.
	:param initialSort: Whether or not to sort the regions (default is `True`).
	
	:type name: str
	:type rgns: list
	:type initialSort: bool
	
	For a region set `RS`, `len(RS)` gives the number of regions, `RS[i]` gives the `i`th region in the set, and `[ r for r in RS ]` lists regions in the set. For sets `A` and `B`, `A+B` gives a sorted region set of regions from both sets, `A|B` gives the merged set, `A&B` the conjunction, and `A^B` the regions of `A` with `B` excluded.
	"""
	
	#__slots__ = ('name', 'regions')
	
	def __init__(self, name, rgns, initialSort = True):
		self.name, self.regions = name, rgns
		if initialSort and len(rgns) != 0:
			self.sort()
	
	def __iter__(self):
		return self.regions.__iter__()
	
	def __getitem__(self, i):
		return self.regions[i]
	
	def __len__(self):
		return len(self.regions)
	
	def __str__(self):
		return 'Region set<%s>'%(self.name)
	
	def _as_dict_(self):
		return {
			'Seq.': [ r.seq for r in self ],
			'Start': [ r.start for r in self ],
			'End': [ r.end for r in self ],
			'Strand': [ '+' if r.strand else '-' for r in self ],
			'Length': [ len(r) for r in self ],
		}
	
	def table(self):
		return nctable(
			'Table: ' + self.__str__(),
			self._as_dict_(),
			align = { 'Seq.': 'l' }
		)
		
	def _repr_html_(self):
		return self.table()._repr_html_()
	
	def __repr__(self):
		return self.table().__repr__()
	
	def __add__(self, other):
		rs = regions('%s + %s'%(self.name, other.name), self.regions + other.regions)
		rs.sort()
		return rs
	
	def __or__(self, other):
		return self.merge(other)
	
	def __and__(self, other):
		return self.intersection(other)
	
	def __xor__(self, other):
		return self.exclusion(other)
	
	def bstr(self):
		"""
		:return: Returns a concatenated list of regions formatted as `seq:start..end (strand)`.
		:rtype: str
		"""
		return ', '.join( r.bstr() for r in self.regions )
	
	# Returns a random division of the sequences
	def randomSplit(self, ratio = 0.5):
		"""
		:param ratio: Ratio for the split, between 0 and 1. For the return touple, `(A, B)`, a ratio of 0 means all the regions are in `B`, and opposite for a ratio of 1.
		:type ratio: float
		
		:return: Returns a random split of the regions into two independent sets, with the given ratio.
		:rtype: regions
		"""
		rgns = self.regions[:]
		isplit = int(len(rgns)*ratio)
		random.shuffle(rgns)
		return regions(self.name + ' (split A)', rgns[:isplit]),\
		       regions(self.name + ' (split B)', rgns[isplit:])
	
	# Returns a filtered subset of regions.
	def filter(self, fltName, flt):
		""" Returns a region set filtered with the input lambda function.
		
		:param fltName: Name of the filter. Appended to the region set name.
		:param flt: Lambda function to be applied to every region, returning `True` if the region is to be included, or otherwise `False`.
		:type fltName: str
		:type flt: lambda
		
		:return: Region set filtered with the lambda function.
		:rtype: regions
		"""
		return regions(self.name + ' (' + fltName + ')', [ r for r in self.regions if flt(r) ])
	
	# Returns the merged region set for this and another set.
	def flatten(self):
		""" Returns a flattened set of regions, with internally overlapping regions merged.
		
		:return: Flattened region set.
		:rtype: regions
		"""
		return self.merge(regions('', []))
	
	# Returns the merged region set for this and another set.
	def merge(self, regions other):
		""" Gets the merged set of regions in this set and another.
		
		:param other: Other region set.
		:type other: regions
		
		:return: Merged region set.
		:rtype: regions
		"""
		cdef list rlist, newlist
		cdef region rgn, lastRgn
		cdef region x
		cdef dict rdict
		cdef list sortedKeys
		rlist = self.regions + other.regions
		if len(rlist) == 0:
			return regions('%s | %s'%(self.name, other.name), [], initialSort = False)
		rdict = {}
		for rgn in rlist:
			if not rgn.seq in rdict:
				rdict[rgn.seq] = [rgn]
			else:
				rdict[rgn.seq].append(rgn)
		sortedKeys = sorted(rdict.keys())
		for cseq in sortedKeys:
			rdict[cseq] = sorted(rdict[cseq], key = lambda x: x.start)
		lastRgn = region(rlist[0].seq, rlist[0].start, rlist[0].end)
		newlist = [ lastRgn ]
		for cseq in sortedKeys:
			for rgn in rdict[cseq]:
				if lastRgn.end+1 < rgn.start or lastRgn.seq != rgn.seq:
					lastRgn = region(rgn.seq, rgn.start, rgn.end)
					newlist.append( lastRgn )
				elif lastRgn.end < rgn.end:
					lastRgn.end = rgn.end
		return regions('%s | %s'%(self.name, other.name), newlist, initialSort = False)
	
	# Returns the intersection of this set and another set.
	def intersection(self, regions other):
		""" Gets the intersection of regions in this set and another.
		
		:param other: Other region set.
		:type other: regions
		
		:return: Intersection region set.
		:rtype: regions
		"""
		cdef list intersection
		cdef list rBs
		cdef region rA, rB
		cdef int iB, firstRelevantIndexB
		intersection = []
		rBs = other.regions
		for rA in self.regions:
			firstRelevantIndexB = 0
			for iB,rB in enumerate(rBs):
				if rB.seq > rA.seq or (rB.seq == rA.seq and rA.end < rB.start):
					break
				if rB.seq < rA.seq or (rB.seq == rA.seq and rB.end < rA.start):
					firstRelevantIndexB = iB
					continue
				intersection.append( region(rA.seq, max(rA.start, rB.start), min(rA.end, rB.end)) )
			if firstRelevantIndexB > 0:
				rBs = rBs[firstRelevantIndexB:]
		return regions('%s & %s'%(self.name, other.name), intersection)
	
	# Returns the excluded region set for this and another set.
	def exclusion(self, regions other):
		""" Gets the regions of this set with regions of another set excluded.
		
		:param other: Other region set.
		:type other: regions
		
		:return: Excluded region set.
		:rtype: regions
		"""
		output = []
		exSet = other.regions
		for i in self.regions:
			oi = region(i.seq, i.start, i.end, i.strand, i.score, i.source, i.feature, group = i.group)
			firstRelevantIndex = 0
			for ei,e in enumerate(exSet):
				if e.seq < oi.seq or (e.seq == oi.seq and e.end < oi.start):
					firstRelevantIndex = ei+1
					continue
				if e.seq > oi.seq or (e.seq == oi.seq and e.start > oi.end):
					break
				if oi.start >= e.start and oi.end <=e.end:
					oi = None
					break
				if e.start > oi.start and e.end < oi.end:
					output.append( region(oi.seq, oi.start, e.start-1, oi.strand, oi.score, oi.source, oi.feature, group = oi.group) )
					oi = region(oi.seq, e.end+1, oi.end, oi.strand, oi.score, oi.source, oi.feature, group = oi.group)
				elif e.start < oi.start:
					if e.end >= oi.start:
						oi.start = e.end+1
				else:
					if e.start <= oi.end:
						oi.end = e.start-1
			exSet = exSet[firstRelevantIndex:]
			if oi:
				output.append( oi )
		rs = regions('%s ^ %s'%(self.name, other.name), output)
		return rs
	
	# Returns the subset of regions from this set that overlap with another set. Note that this assumes that the regions are sorted in both lists, with the class sort() method.
	def overlap(self, regions other):
		""" Gets the regions in this set that overlap with another.
		
		:param other: Other region set.
		:type other: regions
		
		:return: Set of overlapping regions.
		:rtype: regions
		"""
		cdef regions rs
		cdef region rA, rB
		cdef list rAs, rBs, rlist
		cdef int iB, firstRelevantIndexB
		cdef int lenB
		rlist = []
		rAs = self.regions
		rBs = other.regions
		lenB = len(rBs)
		firstRelevantIndexB = 0
		for rA in rAs:
			iB = firstRelevantIndexB
			while True:
				if iB >= lenB:
					break
				rB = rBs[iB]
				if rB.seq > rA.seq:
					break
				if rB.seq < rA.seq:
					iB += 1
					firstRelevantIndexB = iB
					continue
				if rB.end < rA.start:
					iB += 1
					firstRelevantIndexB = iB
					continue
				firstRelevantIndexB = iB
				if rA.end < rB.start:
					break
				rlist.append(rA)
				break
		return regions('%s overlapping with %s'%(self.name, other.name), rlist, initialSort = False)
	
	# Returns the subset of regions from this set that do not overlap with another set. Note that this assumes that the regions are sorted in both lists, with the class sort() method.
	def nonOverlap(self, regions other):
		""" Gets the regions in this set that do not overlap with another.
		
		:param other: Other region set.
		:type other: regions
		
		:return: Set of non-overlapping regions.
		:rtype: regions
		"""
		cdef region rA, rB
		cdef regions rs
		cdef list rBs
		cdef bool over
		cdef int iB, firstRelevantIndexB
		rs = regions('%s not overlapping with %s'%(self.name, other.name), [])
		rBs = other.regions
		for rA in self.regions:
			firstRelevantIndexB = 0
			over = False
			for iB,rB in enumerate(rBs):
				if rB.seq > rA.seq:
					break
				if rB.seq < rA.seq:
					continue
				if rB.end < rA.start:
					continue
				firstRelevantIndexB = iB
				if rA.end < rB.start:
					break
				over = True
				break
			if not over:
				rs.regions.append(rA)
			if firstRelevantIndexB > 0:
				rBs = rBs[firstRelevantIndexB:]
		return rs
	
	# Calculates the overlap sensitivity to another set
	def overlapSensitivity(self, other):
		""" Calculates the overlap sensitivity to another set.
		
		:param other: Other region set.
		:type other: regions
		
		:return: Overlap sensitivity.
		:rtype: float
		"""
		return len(other.overlap(self)) / len(other)
	
	# Calculates the overlap precision to another set
	def overlapPrecision(self, other):
		""" Calculates the overlap precision to another set.
		
		:param other: Other region set.
		:type other: regions
		
		:return: Overlap precision.
		:rtype: float
		"""
		return len(self.overlap(other)) / len(self)
	
	# Calculates the overlap precision to another set
	def nucleotidePrecision(self, other):
		""" Calculates the nucleotide precision to another set.
		
		:param other: Other region set.
		:type other: regions
		
		:return: Nucleotide precision.
		:rtype: float
		"""
		return sum(len(r) for r in self & other) / sum(len(r) for r in self)
	
	# Calculates the overlap precision to another set
	def dummy(self, genome, useSeq = None):
		""" Generates a dummy region set, with the same region lengths but random positions.
		
		:param genome: Genome.
		:param useSeq: Chromosomes to focus on.
		:type genome: genome
		:type useSeq: list
		
		:return: Dummy region set.
		:rtype: regions
		"""
		dr = []
		seqLens = genome.sequenceLengths()
		if useSeq is not None:
			seqLens = { k: seqLens[k] for k in seqLens if k in useSeq }
		genomeLen = sum(seqLens[k] for k in seqLens)
		rs = self.regions
		if useSeq is not None:
			rs = [ r for r in rs if r.seq in useSeq ]
		for r in rs:
			pos = int(random.random() * (genomeLen - len(r)))
			for k in seqLens:
				if pos < seqLens[k]:
					dr.append( region(seq = k, start = pos, end = pos + len(r) - 1) )
					break
				pos -= seqLens[k]
		return regions(self.name + ' (dummy)', dr)
	
	def expected(self, genome, statfun, useSeq = None, repeats = 100):
		""" Calculates an expected statistic by random dummy set creation and averaging.
		
		:param genome: Genome.
		:param statfun: Function to apply to region sets that returns desired statistic.
		:param useSeq: Chromosomes to focus on.
		:param repeats: Number of repeats for calculating statistic.
		:type genome: genome
		:type statfun: Function
		:type useSeq: list
		:type repeats: int
		
		:return: Average statistic.
		:rtype: float
		"""
		stats = [
			statfun(self.dummy(genome = genome, useSeq = useSeq))
			for _ in range(repeats)
		]
		return sum(stats) / len(stats)
	
	# Returns a set of recentered regions, with regions randomly placed within larger regions.
	def recenter(self, long long size):
		""" Gets a set of randomly recentered regions. If the target size is smaller than a given region, a region of the desired size is randomly placed within the region. If the desired size is larger, a region of the desired size is placed with equal center to the center of the region.
		
		:param size: Desired region size.
		:type size: int
		
		:return: Set of randomly recentered regions.
		:rtype: regions
		"""
		cdef list rlist = []
		cdef long long rstart, rend
		cdef region rgn
		for rgn in self.regions:
			if len(rgn) < size:
				#rstart = <long long>( ( rgn.start + rgn.end ) / 2 ) - <long long>(size/2)
				rstart = <long long>( ( rgn.start / 2 ) + ( rgn.end / 2 ) ) - <long long>(size/2)
			elif len(rgn) > size:
				rstart = rgn.start + <long long>( random.uniform( 0, len(rgn)-size ) )
				#rstart += <long long>( random.uniform( 0, len(rgn)-size ) )
			else:
				rstart = rgn.start
			rend = rstart + size - 1
			rlist.append(region(rgn.seq, rstart, rend, strand = rgn.strand, score = rgn.score, source = rgn.source, feature = rgn.feature, group = rgn.group))
		return regions('%s (recentered - %d bp)'%(self.name, size), rlist)
	
	# Returns a set of delta-resized regions.
	def deltaResize(self, sizeDelta):
		""" Gets a set of resized regions. The delta size is subtracted from the start and added to the end.
		
		:param sizeDelta: Desired region size change.
		:type size: int
		
		:return: Set of resized regions.
		:rtype: regions
		"""
		return regions(self.name + ' (%s %d bp)'%('+' if sizeDelta >= 0 else '-', sizeDelta), [
			region(r.seq, r.start - sizeDelta, r.end + sizeDelta, r.strand, r.score, r.source, r.feature, r.group)
			for r in self.regions
			if ( (r.end + sizeDelta) - (r.start - sizeDelta) ) > 0
		])
	
	# Returns a renamed region set.
	def rename(self, newname):
		""":param newname: New name.
		:type newname: str
		
		:return: Renamed region set.
		:rtype: regions
		"""
		return regions(newname, self.regions)
	
	# Sorts regions, first by start coordinate, then by sequence.
	def sort(self):
		""" Sorts the set, first by start coordinate, then by sequence."""
		cdef dict rdict
		cdef list sortedKeys
		cdef region rgn, x
		cdef str cseq
		rdict = {}
		for rgn in self.regions:
			if not rgn.seq in rdict:
				rdict[rgn.seq] = [rgn]
			else:
				rdict[rgn.seq].append(rgn)
		sortedKeys = sorted(rdict.keys())
		self.regions = []
		for cseq in sortedKeys:
			self.regions += sorted(rdict[cseq], key = lambda x: x.start)
	
	# Extracts region sequences from a FASTA file.
	def extractSequences(self, seqs):
		""" Extracts region sequences from a sequence set or stream.
		
		:param seqs: Sequences/sequence stream to extract region sequences from.
		:type seqs: sequences/sequenceStream
		
		:return: Extracted region sequences.
		:rtype: sequences
		"""
		cSeqRegions = []
		ret = sequences('Regions set: ' + self.name + ' - from sequence stream: ' + str(seqs), [])
		if isinstance(seqs, sequences) or isinstance(seqs, list):
			for cseq in seqs:
				ret.sequences += [ sequence(None, cseq.seq[r.start:r.end+1], sourceRegion = r) for r in self.regions if r.seq == cseq.name ]
		elif isinstance(seqs, sequenceStream):
			for cseq in seqs:
				cSeqRegions = [ { 'start':r.start, 'end':r.end, 'seq':sequence(None, '', sourceRegion = r) } for r in self.regions if r.seq == cseq.name ]
				if len(cSeqRegions) == 0: continue
				ret.sequences += [ r['seq'] for r in cSeqRegions ]
				aSeq, bSeq = 0, 0
				for blk in cseq:
					bSeq = aSeq + len(blk)
					# Go through regions. If any overlap with the window, extract.
					# If any precede the window, dump them.
					firstValidRegion = 0
					for i,r in enumerate(cSeqRegions):
						if r['end'] + 1 < aSeq:
							firstValidRegion = i + 1
							continue
						if r['start'] > bSeq:
							break
						r['seq'].seq += blk[ max(r['start']-aSeq, 0) : min(r['end']+1-aSeq, bSeq-aSeq) ]
					if firstValidRegion > 0:
						cSeqRegions = cSeqRegions[firstValidRegion:]
					aSeq = bSeq
		return ret
	
	# Extracts region sequences from a 2bit file.
	def extractSequencesFrom2bit(self, path):
		""" Extracts region sequences from a 2bit file.
		
		:param path: Path to 2bit file.
		:type path: str
		
		:return: Extracted region sequences.
		:rtype: sequences
		"""
		return self.extractSequences( stream2bit(path) )
	
	# Extracts region sequences from a FASTA file.
	def extractSequencesFromFASTA(self, path):
		""" Extracts region sequences from a FASTA file.
		
		:param path: Path to FASTA file.
		:type path: str
		
		:return: Extracted region sequences.
		:rtype: sequences
		"""
		return self.extractSequences( streamFASTA(path) )
	
	# Extracts region sequences from a sequence list, sequence stream, or path.
	def extract(self, src):
		""" Extracts region sequences from a sequence set, sequence stream, or a file by path.
		
		:param src: Sequence set, sequence stream, or sequence file path.
		:type src: sequences/sequenceStream/str
		
		:return: Extracted region sequences.
		:rtype: sequences
		"""
		if isinstance(src, str):
			src = getSequenceStreamFromPath(src)
		if type(src).__name__ == 'genome':
			src = src.sequences
		if isinstance(src, sequences) or isinstance(src, list) or isinstance(src, sequenceStream):
			return self.extractSequences(src)
	
	# Saves regions to GFF file.
	def saveGFF(self, path):
		""" Saves the regions to a General Feature Format (https://www.ensembl.org/info/website/upload/gff.html) file.
		
		:param path: Output file path.
		:type path: str
		"""
		with open(path, 'w') as fOut:
			for r in self.regions:
				fOut.write('%s\t%s\t%s\t%d\t%d\t%f\t%s\t.\t1\n'%(r.seq, r.source, r.feature, r.start, r.end + 1, r.score, '+' if r.strand else '-'))
	
	# Saves regions to BED file.
	def saveBED(self, path):
		""" Saves the regions to a BED (https://www.ensembl.org/info/website/upload/bed.html) file.
		
		:param path: Output file path.
		:type path: str
		"""
		with open(path, 'w') as fOut:
			for r in self.regions:
				fOut.write('%s\t%d\t%d\t%s\t%f\t%s\n'%(r.seq, r.start, r.end, r.feature, r.score, '+' if r.strand else '-'))
	
	# Saves regions to coordinate list file.
	def saveCoordinateList(self, path):
		""" Saves the regions to a coordinate list file (a coordinate per line, with the format `seq:start..end`).
		
		:param path: Output file path.
		:type path: str
		"""
		with open(path, 'w') as fOut:
			for r in self.regions:
				fOut.write('%s:%d..%d\n'%(r.seq, r.start, r.end))
	
	# Prints basic statistics about the set.
	def printStatistics(self):
		"""
		Outputs basic statistics.
		"""
		print('Region set statistics - ' + self.name)
		print(' - Regions: %d'%len(self.regions))
		print(' - Region sequences: %s'%', '.join(sorted(set( r.seq for r in self.regions ))))
		if len(self.regions) > 0:
			print(' - Mean length: %.2f nt (min.: %d nt - max.: %d nt)'%( sum( len(r) for r in self.regions ) / len(self.regions), min( len(r) for r in self.regions), max( len(r) for r in self.regions) ))
			print(' - Total length: %d nt'%( sum( len(r) for r in self.regions ) ))

# Loads regions from a BED file.
def loadBED(path):
	""" Loads regions from a BED (https://www.ensembl.org/info/website/upload/bed.html) file.
	
	:param path: Path to the input file.
	:type path: str
	
	:return: Loaded regions.
	:rtype: regions
	"""
	cdef list rgns
	cdef str l, seq, start, end, source, feature, group
	cdef bool strand
	cdef float score
	cdef list parts
	cdef regions rs
	with open(path, 'r') as f:
		rgns = []
		for l in f:
			if l.startswith('#'): continue
			parts = l.strip().split('\t')
			if len(parts) < 3: raise Exception('BED file syntax error')
			seq, start, end = parts[0:3]
			strand = True
			score = 0.0
			source = '.'
			feature = '.'
			group = '.'
			if len(parts) > 3:
				feature = parts[3]
			if len(parts) > 4:
				score = float(parts[4])
			if len(parts) > 5:
				strand = parts[5] == '+'
			rgns.append( region( seq, int(float(start)), int(float(end)), strand, score, source, feature, group = group ) )
	rs = regions('BED file: ' + path, rgns)
	#rs.sort()
	return rs

# Loads regions from a gzipped BED file.
def loadBEDGZ(path):
	""" Loads regions from a gzipped BED (https://www.ensembl.org/info/website/upload/bed.html) file.
	
	:param path: Path to the input file.
	:type path: str
	
	:return: Loaded regions.
	:rtype: regions
	"""
	cdef list rgns
	cdef str l, y, seq, start, end, source, feature, group
	cdef bool strand
	cdef float score
	cdef list parts
	cdef regions rs
	#with open(path, 'r') as f:
	with gzip.open(path, 'rb') as f:
		rgns = []
		for l in (y.decode('utf-8') for y in f):
			if l.startswith('#'): continue
			parts = l.strip().split('\t')
			if len(parts) < 3: raise Exception('BED file syntax error')
			seq, start, end = parts[0:3]
			strand = True
			score = 0.0
			source = '.'
			feature = '.'
			group = '.'
			if len(parts) > 3:
				feature = parts[3]
			if len(parts) > 4:
				score = float(parts[4])
			if len(parts) > 5:
				strand = parts[5] == '+'
			rgns.append( region( seq, int(float(start)), int(float(end)), strand, score, source, feature, group = group ) )
	rs = regions('BED file: ' + path, rgns)
	return rs

# Loads regions from a GFF file.
def loadGFF(path, dropChr = True):
	""" Loads regions from a General Feature Format (https://www.ensembl.org/info/website/upload/gff.html) file.
	
	:param path: Path to the input file.
	:type path: str
	
	:return: Loaded regions.
	:rtype: regions
	"""
	cdef str x
	cdef list rgns, r
	cdef regions rs
	rgns = []
	with open(path, 'r') as f:
		for x in f:
			if x.startswith('#'): continue
			r = x.strip().split('\t')
			if len(r) != 9: continue
			rgns.append( region( r[0], int(float(r[3])), int(float(r[4]))-1, r[6] != '-', float(r[5]) if r[5] != '.' else 0.0, r[1], r[2], group = r[8], dropChr = dropChr ) )
	rs = regions('GFF file: ' + path, rgns)
	return rs

# Loads regions from a gzipped GFF file.
def loadGFFGZ(path, dropChr = True):
	""" Loads regions from a gzipped General Feature Format (https://www.ensembl.org/info/website/upload/gff.html) file.
	
	:param path: Path to the input file.
	:type path: str
	
	:return: Loaded regions.
	:rtype: regions
	"""
	with gzip.open(path, 'rb') as f:
		raw_gff = [ y for y in (x.split('\t') for x in (y.decode('utf-8') for y in f) if not x.startswith('#')) if len(y)==9 ]
		rgns = [ region( r[0], int(float(r[3])), int(float(r[4]))-1, r[6] != '-', float(r[5]) if r[5] != '.' else 0.0, r[1], r[2], group = r[8], dropChr = dropChr ) for r in raw_gff ]
	rs = regions('GFF GZ file: ' + path, rgns)
	rs.sort()
	return rs

# Loads regions from a coordinate list file.
def loadCoordinateList(path, dropChr = True):
	""" Loads regions from a coordinate list file (one region per line, with the format `seq:start..end`).
	
	:param path: Path to the input file.
	:type path: str
	
	:return: Loaded regions.
	:rtype: regions
	"""
	with open(path, 'r') as f:
		rgns = []
		for l in f:
			e = l.strip().split(':')
			if len(e) != 2:
				raise Exception('Coordinate list syntax error')
			pos = e[1].split('..')
			if len(pos) != 2:
				raise Exception('Coordinate list syntax error')
			rgns.append( region(e[0], int(float(pos[0])), int(float(pos[1])), dropChr = dropChr) )
	rs = regions('Coordinate list file: ' + path, rgns)
	rs.sort()
	return rs

# Generates a prediction overlap precision barplot
def overlapPrecisionBarplot(regionSets, predictionSets, figsize = (8, 8), outpath = None, returnHTML = False, fontsizeLabels = 18, fontsizeLegend = 12, fontsizeAxis = 10, style = 'ggplot', showLegend = True, bboxAnchorTo = (0., -0.15), legendLoc = 'upper left'):
	""" Generates a prediction overlap precision barplot.
	
	:param regionSets: List of validation region sets.
	:param predictionSets: List of prediction region sets.
	:param figsize: Figure size.
	:param outpath: Output path.
	:param returnHTML: If True, an HTML node will be returned.
	:param fontsizeLabels: Size of label font.
	:param fontsizeLegend: Size of legend font.
	:param fontsizeAxis: Size of axis font.
	:param style: Plot style to use.
	:param showLegend: Flag for whether or not to render legend.
	:param bboxAnchorTo: Legend anchor point.
	:param legendLoc: Legend location.
	
	:type regionSets: list
	:type predictionSets: list
	:type figsize: tuple, optional
	:type outpath: str, optional
	:type returnHTML: bool, optional
	:type fontsizeLabels: float, optional
	:type fontsizeLegend: float, optional
	:type fontsizeAxis: float, optional
	:type style: str, optional
	:type showLegend: bool, optional
	:type bboxAnchorTo: tuple, optional
	:type legendLoc: str, optional
	"""
	import matplotlib.pyplot as plt
	import base64
	from io import BytesIO
	from IPython.core.display import display, HTML
	import matplotlib.ticker as mtick
	with plt.style.context(style):
		width = 0.5
		bw = width / (len(predictionSets)-1)
		fig, ax = plt.subplots(figsize = figsize)
		for psi, ps in enumerate(predictionSets):
			x = [
				float(rsi) - (width/2) + (bw*psi)
				for rsi in range(len(regionSets))
			]
			v = [
				100. * len(ps.overlap(rs)) / len(ps)
				for rs in regionSets
			]
			ax.bar(x, v, bw, label = ps.name)
		ax.set_ylabel('Overlap precision', fontsize = fontsizeLabels)
		ax.set_xticks([ float(i) for i in range(len(regionSets)) ])
		ax.set_xticklabels([ rs.name for rs in regionSets ])
		plt.xticks(fontsize = fontsizeLabels, rotation = 0)
		plt.yticks(fontsize = fontsizeAxis, rotation = 0)
		ax.yaxis.set_major_formatter(mtick.PercentFormatter())
		if showLegend:
			ax.legend(bbox_to_anchor = bboxAnchorTo, loc = legendLoc, fontsize = fontsizeLegend, fancybox = True)
		fig.tight_layout()
		if outpath is None:
			bio = BytesIO()
			fig.savefig(bio, format='png')
			plt.close('all')
			encoded = base64.b64encode(bio.getvalue()).decode('utf-8')
			html = '<img src=\'data:image/png;base64,%s\'>'%encoded
			if returnHTML:
				return html
			display(HTML(html))
		else:
			fig.savefig(outpath)
			plt.close('all')

# Generates a prediction overlap recall barplot
def overlapSensitivityBarplot(regionSets, predictionSets, figsize = (8, 8), outpath = None, returnHTML = False, fontsizeLabels = 18, fontsizeLegend = 12, fontsizeAxis = 10, style = 'ggplot', showLegend = True, bboxAnchorTo = (0., -0.15), legendLoc = 'upper left'):
	""" Generates a prediction overlap sensitivity barplot.
	
	:param regionSets: List of validation region sets.
	:param predictionSets: List of prediction region sets.
	:param figsize: Figure size.
	:param outpath: Output path.
	:param returnHTML: If True, an HTML node will be returned.
	:param fontsizeLabels: Size of label font.
	:param fontsizeLegend: Size of legend font.
	:param fontsizeAxis: Size of axis font.
	:param style: Plot style to use.
	:param showLegend: Flag for whether or not to render legend.
	:param bboxAnchorTo: Legend anchor point.
	:param legendLoc: Legend location.
	
	:type regionSets: list
	:type predictionSets: list
	:type figsize: tuple, optional
	:type outpath: str, optional
	:type returnHTML: bool, optional
	:type fontsizeLabels: float, optional
	:type fontsizeLegend: float, optional
	:type fontsizeAxis: float, optional
	:type style: str, optional
	:type showLegend: bool, optional
	:type bboxAnchorTo: tuple, optional
	:type legendLoc: str, optional
	"""
	import matplotlib.pyplot as plt
	import base64
	from io import BytesIO
	from IPython.core.display import display, HTML
	import matplotlib.ticker as mtick
	with plt.style.context(style):
		width = 0.5
		bw = width / (len(predictionSets)-1)
		fig, ax = plt.subplots(figsize = figsize)
		for psi, ps in enumerate(predictionSets):
			x = [
				float(rsi) - (width/2) + (bw*psi)
				for rsi in range(len(regionSets))
			]
			v = [
				100. * len(rs.overlap(ps)) / len(rs)
				for rs in regionSets
			]
			ax.bar(x, v, bw, label = ps.name)
		ax.set_ylabel('Overlap sensitivity', fontsize = fontsizeLabels)
		ax.set_xticks([ float(i) for i in range(len(regionSets)) ])
		ax.set_xticklabels([ rs.name for rs in regionSets ])
		plt.xticks(fontsize = fontsizeLabels, rotation = 0)
		plt.yticks(fontsize = fontsizeAxis, rotation = 0)
		ax.yaxis.set_major_formatter(mtick.PercentFormatter())
		if showLegend:
			ax.legend(bbox_to_anchor = bboxAnchorTo, loc = legendLoc, fontsize = fontsizeLegend, fancybox = True)
		fig.tight_layout()
		if outpath is None:
			bio = BytesIO()
			fig.savefig(bio, format='png')
			plt.close('all')
			encoded = base64.b64encode(bio.getvalue()).decode('utf-8')
			html = '<img src=\'data:image/png;base64,%s\'>'%encoded
			if returnHTML:
				return html
			display(HTML(html))
		else:
			fig.savefig(outpath)
			plt.close('all')

# Generates a prediction nucleotide precision barplot
def nucleotidePrecisionBarplot(regionSets, predictionSets, figsize = (8, 8), outpath = None, returnHTML = False, fontsizeLabels = 18, fontsizeLegend = 12, fontsizeAxis = 10, style = 'ggplot', showLegend = True, bboxAnchorTo = (0., -0.15), legendLoc = 'upper left'):
	""" Generates a prediction nucleotide precision barplot.
	
	:param regionSets: List of validation region sets.
	:param predictionSets: List of prediction region sets.
	:param figsize: Figure size.
	:param outpath: Output path.
	:param returnHTML: If True, an HTML node will be returned.
	:param fontsizeLabels: Size of label font.
	:param fontsizeLegend: Size of legend font.
	:param fontsizeAxis: Size of axis font.
	:param style: Plot style to use.
	:param showLegend: Flag for whether or not to render legend.
	:param bboxAnchorTo: Legend anchor point.
	:param legendLoc: Legend location.
	
	:type regionSets: list
	:type predictionSets: list
	:type figsize: tuple, optional
	:type outpath: str, optional
	:type returnHTML: bool, optional
	:type fontsizeLabels: float, optional
	:type fontsizeLegend: float, optional
	:type fontsizeAxis: float, optional
	:type style: str, optional
	:type showLegend: bool, optional
	:type bboxAnchorTo: tuple, optional
	:type legendLoc: str, optional
	"""
	import matplotlib.pyplot as plt
	import base64
	from io import BytesIO
	from IPython.core.display import display, HTML
	import matplotlib.ticker as mtick
	with plt.style.context(style):
		width = 0.5
		bw = width / (len(predictionSets)-1)
		fig, ax = plt.subplots(figsize = figsize)
		for psi, ps in enumerate(predictionSets):
			x = [
				float(rsi) - (width/2) + (bw*psi)
				for rsi in range(len(regionSets))
			]
			v = [
				100. * sum(len(r) for r in ps &rs) / sum(len(r) for r in ps)
				for rs in regionSets
			]
			ax.bar(x, v, bw, label = ps.name)
		ax.set_ylabel('Nucleotide precision', fontsize = fontsizeLabels)
		ax.set_xticks([ float(i) for i in range(len(regionSets)) ])
		ax.set_xticklabels([ rs.name for rs in regionSets ])
		plt.xticks(fontsize = fontsizeLabels, rotation = 0)
		plt.yticks(fontsize = fontsizeAxis, rotation = 0)
		ax.yaxis.set_major_formatter(mtick.PercentFormatter())
		if showLegend:
			ax.legend(bbox_to_anchor = bboxAnchorTo, loc = legendLoc, fontsize = fontsizeLegend, fancybox = True)
		fig.tight_layout()
		if outpath is None:
			bio = BytesIO()
			fig.savefig(bio, format='png')
			plt.close('all')
			encoded = base64.b64encode(bio.getvalue()).decode('utf-8')
			html = '<img src=\'data:image/png;base64,%s\'>'%encoded
			if returnHTML:
				return html
			display(HTML(html))
		else:
			fig.savefig(outpath)
			plt.close('all')

