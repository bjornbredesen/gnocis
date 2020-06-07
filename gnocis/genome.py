#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from .regions import *
from .curves import curves, variableStepCurve, variableStepCurveValue, fixedStepCurve
from .sequences import *
from .ioputil import nctable

# Represents annotated genes
class gene:
	
	def __init__(self, name, region, synonyms = None, exons = None, CDS = None):
		self.name = name
		self.synonyms = synonyms
		self.region = region
		self.exons = [] if exons is None else exons
		self.CDS = [] if CDS is None else CDS
	
	def __len__(self):
		return self.end+1-self.start
	
	def __str__(self):
		rname = '%s (%s:%d..%d (%s))'%(self.name, self.region.seq, self.region.start, self.region.end, '+' if self.region.strand else '-')
		return 'Gene<%s>'%(rname)
	
	def __repr__(self):
		return self.__str__()


# Represents a genome
class genome:
	
	def __init__(self, name, genes = None, sequences = None, chromosomes = None, annotation = None, annotationPath = None, seqPath = None, seqLens = None):
		self.name = name
		self.genes = genes
		self.sequences = sequences
		self.annotation = annotation
		self.chromosomes = set() if chromosomes is None else chromosomes
		self.annotationPath = annotationPath
		self.seqPath = seqPath
		self.seqLens = seqLens
		self.genesByName = {  }
		self.genesBySyn = {  }
		if genes is not None:
			for g in genes:
				self.genesByName[g.name] = g
				for s in g.synonyms:
					self.genesBySyn[s] = g
	
	def loadEnsemblAnnotationGTFGZ(self, path):
		dat = loadGFFGZ(path)
		annotation = dat
		annotationPath = path
		chromosomes = set(r.seq for r in dat)
		genes = dat.filter('Genes', lambda r: r.feature == 'gene')
		exons = dat.filter('Exons', lambda r: r.feature == 'exon')
		CDS = dat.filter('CDS', lambda r: r.feature == 'CDS')
		getAnnoDict = lambda r: {
			div[0].strip(): div[1].strip()[1:-1]
			for div in (
				q for q in (
					part.strip().split(' ')
					for part in r.group.split(';')
				)
				if len(q) >= 2
			)
		}
		genesByName = {  }
		for r in genes:
			_dict = getAnnoDict(r)
			name = _dict['gene_name']
			g = gene(name = name, region = r, synonyms = [ name, _dict['gene_id'] ])
			genesByName[name] = g
		for r in exons:
			_dict = getAnnoDict(r)
			name = _dict['gene_name']
			genesByName[name].exons.append(r)
		for r in CDS:
			_dict = getAnnoDict(r)
			name = _dict['gene_name']
			genesByName[name].CDS.append(r)
		genes = [ genesByName[k] for k in genesByName ]
		return genome(name = self.name,
			genes = genes,
			sequences = self.sequences,
			chromosomes = self.chromosomes | chromosomes,
			annotation = annotation,
			annotationPath = path,
			seqPath = self.seqPath,
			seqLens = self.seqLens)
	
	def setSequences(self, seqs):
		return genome(name = self.name,
			genes = self.genes,
			sequences = seqs,
			chromosomes = self.chromosomes | set(s.name for s in seqs),
			annotation = self.annotation,
			annotationPath = self.annotationPath,
			seqPath = str(seqs),
			seqLens = None)
	
	def streamFASTA(self, path):
		return self.setSequences(streamFASTA(path))
	
	def streamFASTAGZ(self, path):
		return self.setSequences(streamFASTAGZ(path))
	
	def stream2bit(self, path):
		return self.setSequences(stream2bit(path))
	
	def sequenceLengths(self):
		if self.sequences is None:
			return {}
		if self.seqLens is None:
			self.seqLens = self.sequences.sequenceLengths()
		return self.seqLens
	
	def getCDS(self):
		return regions('CDS', [ r for g in self.genes for r in g.CDS ])
	
	def gene(self, name):
		return self.genesByName[name]
	
	def windows(self, size, step):
		return self.sequences.windows(size = size, step = step)
	
	def windowRegions(self, size, step):
		return self.sequences.windowRegions(size = size, step = step)
	
	def __str__(self):
		return 'Genome<%s; annotation: %s; sequences: %s>'%(self.name,
			'' if self.annotationPath is None else self.annotationPath,
			'' if self.seqPath is None else self.seqPath)
	
	def _as_dict_(self):
		return {
			'Seq.': [ g.region.seq for g in self.genes ],
			'Start': [ g.region.start for g in self.genes ],
			'End': [ g.region.end for g in self.genes ],
			'Strand': [ '+' if g.region.strand else '-' for g in self.genes ],
			'Length': [ len(g.region) for g in self.genes ],
			'Name': [ g.name for g in self.genes ],
			'Synonyms': [ '; '.join(g.synonyms) for g in self.genes ],
			'Exons': [ len(g.exons) for g in self.genes ],
			'CDS': [ len(g.CDS) for g in self.genes ],
		}
	
	def table(self):
		return nctable(
			'Genome: ' + self.__str__(),
			self._as_dict_(),
			align = { 'Seq.': 'l', 'Name': 'l', 'Synonyms': 'l' },
			#strcroplen = 0
		)
	
	def _repr_html_(self):
		return self.table()._repr_html_()
	
	def __repr__(self):
		return self.table().__repr__()

def plotGenomeTracks(tracks, chromosome, coordStart, coordEnd, style = 'ggplot', outpath = None, figsize = None):
	if figsize is None:
		figsize = (10, 1. * len(tracks))
	def coordFmt(t):
		if t >= 1000000:
			return '%.3f Mb'%(t / 1000000.)
		if t >= 1000:
			return '%.3f kb'%(t / 1000.)
		return '%.3f bp'%t
	try:
		import matplotlib.pyplot as plt
		import base64
		from io import BytesIO
		from IPython.core.display import display, HTML
		from matplotlib.collections import PatchCollection
		from matplotlib.patches import Rectangle
		from matplotlib.transforms import ScaledTranslation
		with plt.style.context(style):
			fig, ax = plt.subplots(1, figsize = figsize)
			# Set up axes
			plt.yticks(range(len(tracks)))
			ax.set_xlim(coordStart, coordEnd)
			ax.set_ylim(0, len(tracks))
			ticks = ax.get_xticks()
			ax.set_xticklabels([ coordFmt(t) for t in ticks ], fontsize = 12)
			ax.set_yticklabels([  ])
			setYA = []
			setYB = []
			width = coordEnd - coordStart
			y = 0.
			for rs in tracks:
				yA = y
				if isinstance(rs, genome):
					y += 3.
				elif isinstance(rs, curves):
					y += 3.
					if rs.thresholdValue is not None:
						y += 1.
				else:
					y += 1.
				setYA.append(yA)
				setYB.append(y)
				plt.text(coordStart - width*0.015,
					(y + yA) / 2. + 0.05,
					rs.name,
					verticalalignment = 'top',
					horizontalalignment = 'right',
					color = ax.yaxis.label.get_color(),
					fontsize = 15,
					rotation = 45,
					clip_on = False)
			ax.set_yticks([ 0. ] + setYB)
			ax.yaxis.set_label_coords(-0.3, 1.5)
			plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
				rotation_mode="anchor")
			plt.tick_params(
				axis='y',
				which='both',
				left=False,
				right=False)
			# Shift y axis labels
			dx = -0.1
			dy = 0.2
			offset = ScaledTranslation(dx, dy, fig.dpi_scale_trans)
			for label in ax.yaxis.get_majorticklabels():
				label.set_transform(label.get_transform() + offset)
			# Iterate over tracks
			rmargin = 0.2
			for rsi, ((yA, yB), rs) in enumerate(zip(zip(setYA, setYB), tracks)):
				# Special treatment of gene annotations, with plotting of genes with exons, CDS and names
				height = yB - yA
				if isinstance(rs, genome):
					cgenes = [
						g for g in rs.genes
						if g.region.seq == chromosome\
							 and g.region.end >= coordStart\
							 and g.region.start <= coordEnd
					]
					rects = []
					cy = height / 2.
					for strand in [ False, True ]:
						gh = 0.02
						exonh = 0.2
						ccy = yA + (cy + 0.5 if strand else cy - 0.5)
						rects += [
							Rectangle((g.region.start, ccy - gh), len(g.region), gh*2.)
							for g in cgenes
							if g.region.strand == strand
						]
						rects += [
							Rectangle((r.start, ccy - exonh), len(r), 2*exonh)
							for g in cgenes
							for r in g.exons
							if r.strand == strand
						]
						rects += [
							Rectangle((r.start, ccy - 0.5 + rmargin), len(r), 1. - 2*rmargin)
							for g in cgenes
							for r in g.CDS
							if r.strand == strand
						]
						for g in cgenes:
							if g.region.strand == strand:
								center = (g.region.start + g.region.end)/2.
								if center <= coordStart or center >= coordEnd: continue
								ax.text(center,
									ccy + (0.65 if strand else -0.65),
									g.name,
									horizontalalignment = 'center',
									verticalalignment = 'center')
					pc = PatchCollection(rects, facecolor='C%d'%rsi)
					ax.add_collection(pc)
					continue
				# Special handling of curves
				elif isinstance(rs, curves):
					ccurve = next((c for c in rs if c.seq == chromosome), None)
					if ccurve == None: continue
					if isinstance(ccurve, fixedStepCurve):
						iA = max(int((coordStart - ccurve.span) / ccurve.step), 0)
						iB = max(
								min(int((coordEnd) / ccurve.step), len(ccurve)-1),
							0)
						span = ccurve.span
						step = ccurve.step
						vals = ccurve[iA:iB+1]
						vBase = 0.
						vMin = min(vals)
						vMax = max(vals)
						yBottom = min(vBase, vMin)
						yTop = vMax
						cheight = height
						predHeight = 1.
						if rs.thresholdValue is not None:
							yBottom = min(yBottom, rs.thresholdValue)
							yTop = max(yTop, rs.thresholdValue)
							cheight -= predHeight
						vScale = (cheight - rmargin*2.) / (yTop - yBottom)
						rects = [
							Rectangle(((i + iA) * step,
								yA + rmargin + ((vBase - yBottom) * vScale)),
								span + 1, v * vScale)
							for i, v in enumerate(vals)
						]
						pc = PatchCollection(rects, facecolor='C%d'%rsi)
						ax.add_collection(pc)
						if rs.thresholdValue is not None:
							tthr = yA + rmargin + ((rs.thresholdValue - yBottom) * vScale)
							ax.plot(
								[ coordStart, coordEnd ],
								[ tthr, tthr ],
								linestyle = '--',
								color = 'grey',
								label = 'Expected at random')
							frs = rs.regions().filter('', lambda r: r.seq == chromosome\
								and r.end >= coordStart\
								and r.start <= coordEnd)
							rects = [
								Rectangle((r.start, yB - predHeight + rmargin),
									len(r), predHeight - 2*rmargin)
								for r in frs
							]
							pc = PatchCollection(rects, facecolor='C%d'%rsi)
							ax.add_collection(pc)
					elif isinstance(ccurve, variableStepCurve):
						vals = [
							v for v in ccurve
							if v.start+v.span >= coordStart
							and v.start <= coordEnd
						]
						vBase = 0.
						vMin = min(v.value for v in vals)
						vMax = max(v.value for v in vals)
						yBottom = min(vBase, vMin)
						yTop = vMax
						cheight = height
						predHeight = 1.
						if rs.thresholdValue is not None:
							yBottom = min(yBottom, rs.thresholdValue)
							yTop = max(yTop, rs.thresholdValue)
							cheight -= predHeight
						vScale = (cheight - rmargin*2.) / (yTop - yBottom)
						rects = [
							Rectangle((v.start, yA + rmargin + ((vBase - yBottom) * vScale)),
								v.span + 1, v.value * vScale)
							for v in vals
						]
						pc = PatchCollection(rects, facecolor='C%d'%rsi)
						ax.add_collection(pc)
						if rs.thresholdValue is not None:
							tthr = yA + rmargin + ((rs.thresholdValue - yBottom) * vScale)
							ax.plot(
								[ coordStart, coordEnd ],
								[ tthr, tthr ],
								linestyle = '--',
								color = 'grey',
								label = 'Expected at random')
							frs = rs.regions().filter('', lambda r: r.seq == chromosome\
								and r.end >= coordStart\
								and r.start <= coordEnd)
							rects = [
								Rectangle((r.start, yB - predHeight + rmargin),
									len(r), predHeight - 2*rmargin)
								for r in frs
							]
							pc = PatchCollection(rects, facecolor='C%d'%rsi)
							ax.add_collection(pc)
					continue
				# Simpler handling of region sets
				frs = rs.filter('', lambda r: r.seq == chromosome\
					and r.end >= coordStart\
					and r.start <= coordEnd)
				rects = [
					Rectangle((r.start, yA + rmargin), len(r), height - 2*rmargin)
					for r in frs
				]
				pc = PatchCollection(rects, facecolor='C%d'%rsi)
				ax.add_collection(pc)
			#
			plt.xlabel('Chromosome ' + chromosome, fontsize=18)
			fig.tight_layout()
			if outpath is None:
				bio = BytesIO()
				fig.savefig(bio, format='png')
				plt.close('all')
				encoded = base64.b64encode(bio.getvalue()).decode('utf-8')
				html = '<img src=\'data:image/png;base64,%s\'>'%encoded
				display(HTML(html))
			else:
				fig.savefig(outpath)
				plt.close('all')
	#
	except ImportError as err:
		raise err


