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
from .regions import regions, region
from .ioputil import nctable
#from libcpp cimport bool
#from .sequences cimport *
#from .ioputil import nctable


############################################################################
# Fixed step curve

class fixedStepCurve:
	
	def __init__(self, seq, start, span, step, values = None, dropChr = True):
		if dropChr:
			if seq.lower().startswith('chr'):
				seq = seq[3:]
		self.seq = seq
		self.start = start
		self.span = span
		self.step = step
		self.values = [] if values is None else values
	
	def __iter__(self):
		return self.values.__iter__()
	
	def __getitem__(self, i):
		return self.values[i]
	
	def __len__(self):
		return len(self.values)
	
	def __str__(self):
		return 'Fixed step curve<%s; start = %d; span = %d; step = %d>'%(self.seq, self.start, self.span, self.step)
	
	def __repr__(self):
		return self.__str__()
	
	def fillGaps(self):
		return fixedStepCurve(seq = self.seq, start = self.start, span = self.span, step = self.step, values = self.values[:])
	
	def regions(self, threshold):
		rgns = []
		rA = 0
		for v in self:
			rB = rA + self.span
			if v >= threshold:
				if len(rgns) == 0 or rgns[-1].end < rA:
					rgns.append( region(seq = self.seq, start = rA, end = rB, score = v) )
				else:
					rgns[-1].end = rB
					rgns[-1].score = max(rgns[-1].score, v)
			rA += self.step
		return regions('', rgns)


############################################################################
# Fixed step curve

class variableStepCurveValue:
	
	def __init__(self, start, span, value):
		self.start = start
		self.span = span
		self.value = value
	
	def __str__(self):
		return 'Variable step curve value<start = %d; span = %d; value = %f>'%(self.start, self.span, self.value)
	
	def __repr__(self):
		return self.__str__()


class variableStepCurve:
	
	def __init__(self, seq, values = None, dropChr = True):
		if dropChr:
			if seq.lower().startswith('chr'):
				seq = seq[3:]
		self.seq = seq
		self.values = [] if values is None else values
	
	def __iter__(self):
		return self.values.__iter__()
	
	def __getitem__(self, i):
		return self.values[i]
	
	def __len__(self):
		return len(self.values)
	
	def __str__(self):
		return 'Variable step curve<%s>'%(self.seq)
	
	def __repr__(self):
		return self.__str__()
	
	def fillGaps(self):
		return variableStepCurve(seq = self.seq, values = [
			variableStepCurveValue(start = vA.start, span = vB.start - vA.start,
				value = vA.value)
			for vA, vB in zip(self.values, self.values[1:] + self.values[-1:])
		])
	
	def regions(self, threshold):
		return regions('', [
			region(seq = self.seq, start = v.start, end = v.start + v.span, score = v.value)
			for v in self
			if v.value >= threshold
		]).flatten()


###########################################################################
# Curves

def parseWigTrackHeader(hdr):
	div = hdr.find(' ')
	typeName = hdr[:div]
	remainder = hdr[div+1:].strip()
	spec = { 'head': typeName }
	while True:
		eql = remainder.find('=')
		if eql == -1: break
		vName = remainder[:eql].strip()
		rm = remainder[eql+1:].strip()
		if rm.startswith('"'):
			i = rm.find('"', 1)
			vValue = rm[1:i]
			remainder = rm[i+1:]
		else:
			i = rm.find(' ')
			vValue = rm[:i] if i != -1 else rm
			remainder = rm[i:]
		spec[vName] = vValue
	return spec


class curves:
	
	def __init__(self, name, curves = None, threshold = None):
		self.name = name
		self.thresholdValue = threshold
		self.curves = [] if curves is None else curves
		if curves is not None:
			self.sort()
	
	def __iter__(self):
		return self.curves.__iter__()
	
	def __getitem__(self, i):
		return self.curves[i]
	
	def __len__(self):
		return len(self.curves)
	
	def __str__(self):
		return 'Curve set<%s>'%(self.name)
	
	def _as_dict_(self):
		if self.thresholdValue is not None:
			rs = self.regions()
			return {
				'Seq.': [ c.seq for c in self ],
				'Values': [ len(c) for c in self ],
				'Regions': [ len(rs.filter('', lambda r: r.seq == c.seq)) for c in self ],
			}
		return {
			'Seq.': [ c.seq for c in self ],
			'Values': [ len(c) for c in self ],
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
	
	def regions(self):
		""":return: Thresholded regions.
		:rtype: regions
		"""
		return regions(self.name, [
			r
			for c in self
			for r in c.regions(self.thresholdValue)
		])
	
	def threshold(self, threshold):
		""":param threshold: Threshold.
		:type threshold: float
		
		:return: Copy of curves with threshold set.
		:rtype: curves
		"""
		return curves(self.name + ' (threshold = %f)'%threshold, self.curves, threshold = threshold)
	
	def fillGaps(self):
		""":return: Copy of curves with gaps in variable step curves filled.
		:rtype: curves
		"""
		return curves(self.name + ' (gaps filled)', [ c.fillGaps() for c in self ], threshold = self.thresholdValue)
	
	def rename(self, newname):
		""":param newname: New name.
		:type newname: str
		
		:return: Renamed curve set.
		:rtype: curves
		"""
		return curves(newname, self.curves, threshold = self.thresholdValue)
	
	def sort(self):
		""" Sorts variable step curves by start coordinate, and curves by name."""
		for c in self.curves:
			if isinstance(c, variableStepCurve):
				c.values = sorted(c.values, key = lambda v: v.start)
		self.curves = sorted(self.curves, key = lambda c: c.seq)
	
	def saveWIG(self, path):
		""" Saves curves to Wiggle-format file.
		
		:param path: Path.
		:type path: str
		"""
		with open(path, 'w') as f:
			for c in self.curves:
				if isinstance(c, fixedStepCurve):
					f.write('fixedStep chrom=%s start=%d step=%d span=%d\n'%(c.seq, c.start, c.step, c.span))
					for v in c:
						f.write(str(v) + '\n')
				elif isinstance(c, variableStepCurve):
					spans = set(v.span for v in c)
					if len(spans) == 1:
						f.write('variableStep chrom=%s span=%d\n'%(c.seq, list(spans)[0]))
						for v in c:
							f.write('%d '%v.start + str(v.value) + '\n')
					else:
						f.write('%s %d %d %s\n'%(v.seq, v.start, v.start + v.span, str(v.value)))

def loadWIGFromLines(name, lines, dropChr = True):
	_curves = []
	ccurve = None
	span = None
	trackSpecified = False
	metadata = None
	for line in lines:
		if line.startswith('track '):
			# Track metadata
			metadata = parseWigTrackHeader(line)
		elif line.startswith('fixedStep '):
			# Fixed step track
			hdr = parseWigTrackHeader(line)
			ccurve = fixedStepCurve(
				seq = hdr['chrom'],
				start = int(hdr['start']),
				span = int(hdr['span']) if 'span' in hdr.keys() else int(hdr['step']),
				step = int(hdr['step']),
				dropChr = dropChr)
			_curves.append(ccurve)
			trackSpecified = True
		elif line.startswith('variableStep '):
			# Variable step track
			hdr = parseWigTrackHeader(line)
			ccurve = variableStepCurve(
				seq = hdr['chrom'],
				dropChr = dropChr)
			_curves.append(ccurve)
			span = int(hdr['span']) if 'span' in hdr.keys() else None
			trackSpecified = True
		else:
			lstrip = line.strip()
			if len(lstrip) == 0: continue
			# Values
			if not trackSpecified:
				# BedGraph
				p = lstrip.split()
				if len(p) != 4:
					raise Exception("Wiggle parsing error: Invalid BedGraph line")
				seqName = p[0]
				if dropChr:
					if seqName.lower().startswith('chr'):
						seqName = seqName[3:]
				if ccurve is not None and ccurve.seq != seqName:
					ccurve = next((c for c in _curves if c.seq == seqName), None)
				if ccurve is None:
					ccurve = variableStepCurve(
						seq = p[0],
						dropChr = dropChr)
					_curves.append(ccurve)
				ccurve.values.append(
					variableStepCurveValue(
						start = int(p[1]),
						span = int(p[2]) - int(p[1]),
						value = float(p[3])
					))
			elif isinstance(ccurve, fixedStepCurve):
				# Fixed step
				ccurve.values.append(float(lstrip))
			elif isinstance(ccurve, variableStepCurve):
				# Variable step
				p = lstrip.split()
				ccurve.values.append(
					variableStepCurveValue(
						start = int(p[0]),
						span = span,
						value = float(p[1])
					))
	return curves(name, _curves)

def loadWIG(path, dropChr = True):
	""" Loads curves from a Wiggle (https://www.ensembl.org/info/website/upload/wig.html) file.
	
	:param path: Path to the input file.
	:type path: str
	
	:return: Loaded curves.
	:rtype: curves
	"""
	pathName = path.split('/')[-1]
	with open(path, 'r') as f:
		lines = (line for line in f)
		return loadWIGFromLines('WIG file: ' + pathName, lines, dropChr = dropChr)

def loadWIGGZ(path, dropChr = True):
	""" Loads curves from a gzipped Wiggle (https://www.ensembl.org/info/website/upload/wig.html) file.
	
	:param path: Path to the input file.
	:type path: str
	
	:return: Loaded curves.
	:rtype: curves
	"""
	pathName = path.split('/')[-1]
	with gzip.open(path, 'rb') as f:
		lines = (line for line in (y.decode('utf-8') for y in f))
		return loadWIGFromLines('WIG GZ file: ' + pathName, lines, dropChr = dropChr)

