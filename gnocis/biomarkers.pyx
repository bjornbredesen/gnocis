# distutils: language=c++
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from __future__ import division
import random
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from .regions cimport *


############################################################################
# Biomarker sets

# Represents biomarkers
cdef class biomarkers:
	"""
	The `biomarkers` class represents a set of biomarker regions. Up to multiple regions marking a phenomenon of interest can be registered in a `biomarker` object. 
	
	:param name: Name of the biomarker set.
	:param regionSets: Name of the biomarker set. Defaults to [].
	:param positiveThreshold: Threshold for Highly BioMarker-Enriched (HBME) loci. Defaults to -1.
	:param negativeThreshold: Threshold for Lowly BioMarker-Enriched (LBME) loci. Defaults to -1.
	
	:type name: str
	:type regionSets: list, optional
	:type positiveThreshold: int, optional
	:type negativeThreshold: int, optional
	
	After constructing a `biomarkers` object, for a set of biomarkers `BM`, `len(BM)` gives the number of markers, `BM['FACTOR_NAME']` gives the merged set of regions for factor `FACTOR_NAME`, and `[ x for x in BM ]` gives the list of merged regions per biomarker contained in `BM`.
	"""
	
	def __init__(self, name, regionSets = [], positiveThreshold = -1, negativeThreshold = -1):
		self.name = name
		self.positiveThreshold = positiveThreshold
		self.negativeThreshold = negativeThreshold
		self.biomarkers = {}
		if len(regionSets) > 0:
			for rs in regionSets:
				self.addBiomarker(rs)
	
	def __len__(self):
		return len(self.biomarkers)
	
	def __getitem__(self, bm):
		return self.biomarkers[bm]['regions']
	
	def __iter__(self):
		return ( self.biomarkers[bm]['regions'] for bm in self.biomarkers )
	
	def __str__(self):
		return 'Biomarker set<%s (%s)>'%(self.name, '; '.join( '%s (%d regions - %d sets)'%(bm, len(self.biomarkers[bm]['regions']), self.biomarkers[bm]['regionSets']) for bm in sorted( self.biomarkers ) ))
	
	def __repr__(self):
		return self.__str__()
	
	# Adds a biomarker to the set, by name and region set.
	# The name of the region set is taken as the name of the biomarker.
	def addBiomarker(self, regions rs):
		""" Adds a biomarker to the set, by name and region set. The name of the region set is taken as the name of the biomarker.
		
		:param rs: Region set to add as biomarker.
		:type rs: regions
		"""
		if not rs.name in self.biomarkers.keys():
			self.biomarkers[rs.name] = { 'name': rs.name, 'regionSets': 1, 'regions': rs }
		else:
			self.biomarkers[rs.name]['regionSets'] += 1
			self.biomarkers[rs.name]['regions'] = self.biomarkers[rs.name]['regions'].getMerged(rs).getRenamed(rs.name)
	
	# Returns the biomarker spectrum (subsets enriched in N biomarkers, for every valid N) as a dictionary.
	def getRegionSetBiomarkerSpectrum(self, regions rs):
		""" Returns the biomarker spectrum. This is a dictionary with numerical keys, containing subsets enriched in N biomarkers, for every valid N as key.
		
		:param rs: Region set to add as biomarker.
		:type rs: regions
		
		:return: Biomarker spectrum
		:rtype: dict
		"""
		cdef region r
		cdef regions enriched
		cdef str m
		cdef int t
		for r in rs:
			r.markers = set()
		for m in self.biomarkers.keys():
			enriched = rs.getOverlap(self.biomarkers[m]['regions'])
			for r in enriched:
				r.markers.add(m)
		return {
			t: regions('%s - enriched in %d/%d biomarkers (%s)'%(rs.name, t, len(self.biomarkers.keys()), ', '.join(m for m in self.biomarkers.keys())), [ r for r in rs if len(r.markers) == t ])
			for t in range(len(self.biomarkers.keys())+1)
		}
	
	# Gets highly biomarker-enriched regions of a set.
	def getHBMEs(self, regions rs, int threshold = -1):
		""" Gets highly biomarker-enriched regions of a set. This is determined either by an optional `threshold` argument, or, if unspecified, by the `positiveThreshold` member. The resulting set is the merged biomarker spectrum for enrichment levels >= to the threshold.
		
		:param rs: Region set to extract enriched subset of.
		:type rs: regions
		
		:return: Highly BioMarker-Enriched regions (HBMEs).
		:rtype: regions
		"""
		cdef dict BME
		cdef int nBME
		cdef region r
		cdef regions cBME
		cdef list rlist = []
		if threshold == -1: threshold = self.positiveThreshold
		BME = self.getRegionSetBiomarkerSpectrum(rs)
		for nBME in range(threshold, len(self.biomarkers)+1):
			cBME = BME[nBME]
			rlist += cBME.regions
		return regions('', rlist).getMerged(regions('', [])).getRenamed('%s (%s HBME)'%(rs.name, self.name))
	
	# Gets lowly biomarker-enriched regions of a set.
	def getLBMEs(self, regions rs, int threshold = -1):
		""" Gets lowly biomarker-enriched regions of a set. This is determined either by an optional `threshold` argument, or, if unspecified, by the `negativeThreshold` member. The resulting set is the merged biomarker spectrum for enrichment levels <= to the threshold.
		
		:param rs: Region set to extract enriched subset of.
		:type rs: regions
		
		:return: Highly BioMarker-Enriched regions (HBMEs)
		:rtype: regions
		"""
		cdef dict BME
		cdef int nBME
		cdef region r
		cdef regions cBME
		cdef list rlist = []
		if threshold == -1: threshold = self.negativeThreshold
		BME = self.getRegionSetBiomarkerSpectrum(rs)
		for nBME in range(0, threshold+1):
			cBME = BME[nBME]
			rlist += cBME.regions
		return regions('', rlist).getMerged(regions('', [])).getRenamed('%s (%s LBME)'%(rs.name, self.name))


