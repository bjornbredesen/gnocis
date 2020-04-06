#!/usr/bin/python3
# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

from .regions import *
from .sequences import *
from .ioputil import nctable

# Represents annotated genes
class gene:
	
	def __init__(self, name, rgn, synonyms = None, exons = None, CDS = None):
		self.name = name
		self.synonyms = synonyms
		self.rgn = rgn
		self.exons = [] if exons is None else exons
		self.CDS = [] if CDS is None else CDS
	
	def __len__(self):
		return self.end+1-self.start
	
	def __str__(self):
		rname = '%s (%s:%d..%d (%s))'%(self.name, self.rgn.seq, self.rgn.start, self.rgn.end, '+' if self.strand else '-')
		return 'Gene<%s>'%(rname)
	
	def __repr__(self):
		return self.__str__()


# Represents a genome
class genome:
	
	def __init__(self, name, annotationPath = None, seqPath = None, chromosomes = None):
		self.name = name
		self.annotationPath = annotationPath
		self.seqPath = seqPath
		self.chromosomes = chromosomes
	
	def loadEnsemblAnnotationGTFGZ(self, path):
		dat = loadGFFGZ(path)
		self.annotation = dat
		self.annotationPath = path
		self.chromosomes = set(r.seq for r in dat)
		genes = dat.getFiltered('Genes', lambda r: r.feature == 'gene')
		exons = dat.getFiltered('Exons', lambda r: r.feature == 'exon')
		CDS = dat.getFiltered('CDS', lambda r: r.feature == 'CDS')
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
		self.genes = [  ]
		self.genesByName = {  }
		self.genesBySyn = {  }
		for r in genes:
			_dict = getAnnoDict(r)
			name = _dict['gene_name']
			g = gene(name = name, rgn = r, synonyms = [ name, _dict['gene_id'] ])
			self.genesByName[name] = g
			self.genesBySyn[name] = g
			self.genesBySyn[_dict['gene_id']] = g
		for r in exons:
			_dict = getAnnoDict(r)
			name = _dict['gene_name']
			self.genesByName[name].exons.append(r)
		for r in CDS:
			_dict = getAnnoDict(r)
			name = _dict['gene_name']
			self.genesByName[name].CDS.append(r)
		self.genes = [ self.genesByName[k] for k in self.genesByName ]
	
	def __str__(self):
		return 'Genome<%s; annotation: %s; sequences: %s>'%(self.name,
			'' if self.annotationPath is None else self.annotationPath,
			'' if self.seqPath is None else self.seqPath)
	
	def _as_dict_(self):
		return {
			'Seq.': [ r.rgn.seq for r in self.genes ],
			'Start': [ r.rgn.start for r in self.genes ],
			'End': [ r.rgn.end for r in self.genes ],
			'Strand': [ '+' if r.rgn.strand else '-' for r in self.genes ],
			'Length': [ len(r.rgn) for r in self.genes ],
			'Name': [ r.name for r in self.genes ],
			'Synonyms': [ '; '.join(r.synonyms) for r in self.genes ],
			'Exons': [ len(r.exons) for r in self.genes ],
			'CDS': [ len(r.CDS) for r in self.genes ],
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
			ax.set_yticklabels([ rs.name for rs in tracks ], fontsize = 16)
			setY = [ 0. ]
			y = 0.
			for rs in tracks:
				if isinstance(rs, genome):
					y += 3.
				else:
					y += 1.
				setY.append(y)
			ax.set_yticks(setY)
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
			rmargin = 0.1
			for rsi, (y, rs) in enumerate(zip(setY, tracks)):
				# Special treatment of gene annotations, with plotting of genes with exons, CDS and names
				if isinstance(rs, genome):
					cgenes = [
						g for g in rs.genes
						if g.rgn.seq == chromosome\
							 and g.rgn.end >= coordStart\
							 and g.rgn.start <= coordEnd
					]
					rects = []
					cy = 3. / 2.
					for strand in [ False, True ]:
						gh = 0.02
						exonh = 0.25
						ccy = y + (cy + 0.5 if strand else cy - 0.5)
						rects += [
							Rectangle((g.rgn.start, ccy - gh), len(g.rgn), gh*2.)
							for g in cgenes
							if g.rgn.strand == strand
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
							if g.rgn.strand == strand:
								plt.text((g.rgn.start + g.rgn.end)/2.,
									ccy + (0.65 if strand else -0.65),
									g.name,
									horizontalalignment = 'center',
									verticalalignment = 'center')
					
					pc = PatchCollection(rects, facecolor='C%d'%rsi)
					ax.add_collection(pc)
					continue
				# Simpler handling of region sets
				frs = rs.getFiltered('', lambda r: r.seq == chromosome\
					and r.end >= coordStart\
					and r.start <= coordEnd)
				rects = [
					Rectangle((r.start, y + rmargin), len(r), 1. - 2*rmargin)
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


