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
from .common import mean, CI

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


############################################################################
# Genome plotting

def drawRoundBox(ax, start, end, width, y, height, fc, ec, maxYB, rlabel = None, clip_on = True):
	import matplotlib.pyplot as plt
	from matplotlib.patches import FancyBboxPatch
	w = end - start + 1
	rs = min(w * 0.5 * width * 0.00001, 500. * width * 0.00001)
	patch = FancyBboxPatch((start, y),
		w, height,
		boxstyle="round,rounding_size=" + str(rs),
		mutation_aspect = 2.*maxYB/width,
		fc = fc,
		ec = ec,
		clip_on = clip_on)
	ax.add_patch(patch)
	if rlabel is not None:
		plt.text(end + width*0.015,
					y + height*.5,
					rlabel,
					verticalalignment = 'center',
					horizontalalignment = 'left',
					color = ax.yaxis.label.get_color(),
					fontsize = 10,
					clip_on = False)
	return patch

def drawBox(ax, start, end, width, y, height, fc, ec, maxYB, rlabel = None, clip_on = True):
	import matplotlib.pyplot as plt
	from matplotlib.patches import FancyBboxPatch
	w = end - start + 1
	rs = min(w * width * 0.00001, 500. * width * 0.00001)
	patch = FancyBboxPatch((start, y),
		w, height,
		boxstyle="square",
		mutation_aspect = 2.*maxYB/width,
		fc = fc,
		ec = ec,
		clip_on = clip_on)
	ax.add_patch(patch)
	if rlabel is not None:
		plt.text(end + width*0.015,
					y + height*.5,
					rlabel,
					verticalalignment = 'center',
					horizontalalignment = 'left',
					color = ax.yaxis.label.get_color(),
					fontsize = 10,
					clip_on = False)
	return patch

def drawGeneBody(ax, start, end, width, y, height, maxYB, rlabel = None, clip_on = True):
	colBody = [ 0.75, 0.75, 0.75 ]
	colInvisible = [ 0., 0., 0., 0. ]
	return drawRoundBox(ax = ax, start = start, end = end, width = width,
		y = y, height = height, maxYB = maxYB,
		fc = colBody + [0.4],
		ec = colInvisible,
		rlabel = rlabel,
		clip_on = clip_on)

def drawGeneOutline(ax, start, end, width, y, height, maxYB, clip_on = True):
	colBorder = [ 0.1, 0.1, 0.1, 1. ]
	colInvisible = [ 0., 0., 0., 0. ]
	drawRoundBox(ax = ax, start = start, end = end, width = width,
		y = y, height = height, maxYB = maxYB,
		fc = colInvisible,
		ec = colBorder,
		clip_on = clip_on)

def drawGeneExon(ax, start, end, y, width, height, maxYB, rlabel = None, clip_on = True):
	colExons = [ 0.45, 0.45, 0.45 ]
	colInvisible = [ 0., 0., 0., 0. ]
	drawBox(ax = ax, start = start, end = end, width = width,
		y = y, height = height, maxYB = maxYB,
		fc = colExons + [1.],
		ec = colInvisible,
		rlabel = rlabel,
		clip_on = clip_on)

def drawGeneCDS(ax, start, end, y, width, height, maxYB, rlabel = None, clip_on = True):
	colCDS = [ 0.1, 0.1, 0.1 ]
	colInvisible = [ 0., 0., 0., 0. ]
	drawBox(ax = ax, start = start, end = end, width = width,
		y = y, height = height, maxYB = maxYB,
		fc = colCDS + [1.],
		ec = colInvisible,
		rlabel = rlabel,
		clip_on = clip_on)

#------------------------------------

class GenomePlotTrack:
	def __init__(self):
		pass
	def getHeight(self):
		return 0.0
	def getName(self):
		return ''
	def draw(self, ax, chromosome, coordStart, coordEnd, width, rmargin, maxYB, colour):
		pass

class GenomePlotTrackRegions(GenomePlotTrack):
	def __init__(self, rs):
		self.rs = rs
	def getHeight(self):
		return 1.0
	def getName(self):
		return self.rs.name
	def draw(self, ax, chromosome, coordStart, coordEnd, width, rmargin, maxYB, colour):
		frs = self.rs.filter('', lambda r: r.seq == chromosome\
			and r.end >= coordStart\
			and r.start <= coordEnd)
		height = self.yB - self.yA
		for r in frs:
			drawRoundBox(ax = ax, start = r.start, end = r.end, width = width,
				y = self.yA + rmargin, height = height - 2*rmargin, maxYB = maxYB,
				fc = colour[:3] + [0.4],
				ec = [ c*0.5 for c in colour[:3] ] + [1.])

class GenomePlotTrackGenome(GenomePlotTrack):
	def __init__(self, rs):
		self.rs = rs
	def getHeight(self):
		return 3.0
	def getName(self):
		return self.rs.name
	def draw(self, ax, chromosome, coordStart, coordEnd, width, rmargin, maxYB, colour):
		height = self.yB - self.yA
		cgenes = [
			g for g in self.rs.genes
			if g.region.seq == chromosome\
				 and g.region.end >= coordStart\
				 and g.region.start <= coordEnd
		]
		cy = height / 2.
		geneheight = 1. - 2.*rmargin
		colInvisible = [ 0., 0., 0., 0. ]
		# Legend: Gene body
		drawGeneBody(ax = ax, start = coordEnd + width*0.01, end = coordEnd + width*0.01 + 3. * 0.006 * width, width = width, y = self.yB - 0.1 - geneheight, height = geneheight, maxYB = maxYB, rlabel = 'Gene body', clip_on = False)
		drawGeneOutline(ax = ax, start = coordEnd + width*0.01, end = coordEnd + width*0.01 + 3. * 0.006 * width, width = width, y = self.yB - 0.1 - geneheight, height = geneheight, maxYB = maxYB, clip_on = False)
		# Legend: Gene exon
		drawGeneExon(ax = ax, start = coordEnd + width*0.01, end = coordEnd + width*0.01 + 3. * 0.006 * width, width = width, y = self.yB - 0.1 - geneheight - 0.1 - geneheight, height = geneheight, maxYB = maxYB, rlabel = 'Exon', clip_on = False)
		# Legend: Gene CDS
		drawGeneCDS(ax = ax, start = coordEnd + width*0.01, end = coordEnd + width*0.01 + 3. * 0.006 * width, width = width, y = self.yB - 0.1 - geneheight - 0.1 - geneheight - 0.1 - geneheight, height = geneheight, maxYB = maxYB, rlabel = 'CDS', clip_on = False)
		#
		for strand in [ False, True ]:
			gh = 0.02
			exonh = 0.2
			ccy = self.yA + (cy + 0.5 if strand else cy - 0.5)
			for g in cgenes:
				if g.region.strand != strand: continue
				# Gene body
				drawGeneBody(ax = ax, start = g.region.start, end = g.region.end, width = width, y = ccy - geneheight*.5, height = geneheight, maxYB = maxYB)
				# Exons
				for r in g.exons:
					drawGeneExon(ax = ax, start = r.start, end = r.end, width = width, y = ccy - geneheight*.5, height = geneheight, maxYB = maxYB)
				# CDS
				for r in g.CDS:
					drawGeneCDS(ax = ax, start = r.start, end = r.end, width = width, y = ccy - geneheight*.5, height = geneheight, maxYB = maxYB)
				#
				drawGeneOutline(ax = ax, start = g.region.start, width = width, end = g.region.end, y = ccy - geneheight*.5, height = geneheight, maxYB = maxYB)
			for g in cgenes:
				if g.region.strand == strand:
					center = (g.region.start + g.region.end)/2.
					if center <= coordStart or center >= coordEnd: continue
					ax.text(center,
						ccy + (0.65 if strand else -0.65),
						g.name,
						horizontalalignment = 'center',
						verticalalignment = 'center')

class GenomePlotTrackCurve(GenomePlotTrack):
	def __init__(self, rs):
		self.rs = rs
	def getHeight(self):
		if self.rs.thresholdValue is not None:
			return 3.0
		else:
			return 1.0
	def getName(self):
		return self.rs.name
	def draw(self, ax, chromosome, coordStart, coordEnd, width, rmargin, maxYB, colour):
		import matplotlib.pyplot as plt
		from matplotlib.patches import Rectangle, Polygon
		from matplotlib.lines import Line2D
		colInvisible = [ 0., 0., 0., 0. ]
		height = self.yB - self.yA
		ccurve = next((c for c in self.rs if c.seq == chromosome), None)
		if ccurve == None: return
		# Rasterize - logic required depends on curve type
		if isinstance(ccurve, fixedStepCurve):
			iA = max(int((coordStart - ccurve.span) / ccurve.step), 0)
			iB = max(
					min(int((coordEnd) / ccurve.step), len(ccurve)-1),
				0)
			span = ccurve.span
			step = ccurve.step
			vals = ccurve[iA:iB+1]
			#
			res = 2048
			srcXA = iA * ccurve.step #coordStart
			srcXB = iB * ccurve.step #coordEnd
			scale = res / (srcXB - srcXA)
			ret = [ None for _ in range(res) ]
			for i, v in enumerate(vals):
				dstXA = int(min(max(
					i * ccurve.step * res / ((iB - iA) * ccurve.step),
					0),
					res-1))
				dstXB = int(min(max(
					res * (i + ccurve.span / ccurve.step) / (iB - iA),
					0),
					res-1))
				for x in range(dstXA, dstXB+1):
					Q = (ccurve.step * (iA + (iB - iA) * float(x) / res), v)
					if ret[x] is None or Q[1] > ret[x][1]:
						ret[x] = Q
			raster = [ v for v in ret if v is not None ]
		elif isinstance(ccurve, variableStepCurve):
			vals = [
				v for v in ccurve
				if v.start+v.span >= coordStart
				and v.start <= coordEnd
			]
			#
			res = 2048
			srcXA = coordStart
			srcXB = coordEnd
			scale = res / (srcXB - srcXA)
			ret = [ None for _ in range(res) ]
			for v in vals:
				dstXA = int(min(max((v.start - srcXA) * scale, 0), res-1))
				dstXB = int(min(max((v.start + v.span - srcXA) * scale, 0), res-1))
				for x in range(dstXA, dstXB+1):
					ret[x] = (srcXA + (float(x) / scale), v.value)
			raster = [ v for v in ret if v is not None ]
		# Plot curve
		vBase = 0.
		vMin = min(y for x, y in raster)
		vMax = max(y for x, y in raster)
		yBottom = min(vBase, vMin)
		yTop = vMax
		cheight = height
		predHeight = 1.
		def V2Y(v):
			return self.yA + rmargin + ((v + vBase - yBottom) * vScale)
		if self.rs.thresholdValue is not None:
			yBottom = min(yBottom, self.rs.thresholdValue)
			yTop = max(yTop, self.rs.thresholdValue)
			cheight -= predHeight
		vScale = (cheight - rmargin*2.) / (yTop - yBottom)
		polypts = [ ]
		polypts += [
			(x, V2Y(max(y, vBase)))
			for x, y in raster ]
		polypts += reversed([
			(x, V2Y(min(y, vBase)))
			for x, y in raster ])
		poly = Polygon(
			polypts,
			fc = colour[:3] + [0.4],
			ec = colInvisible)
		ax.add_patch(poly)
		ax.add_line(Line2D([ x for x, y in raster ],
			[ V2Y(y) for x, y in raster ],
			linewidth = 0.8,
			color = [ c*0.75 for c in colour[:3] ] + [0.75]))
		# Legend
		lyA = V2Y(vMin)
		lyB = V2Y(vMax)
		ax.add_patch(Rectangle(
			(coordEnd, lyA),
			1. * 0.006 * width,
			lyB-lyA,
			fc = (0.2, 0.2, 0.2, 1.),
			ec = colInvisible,
			clip_on = False
		))
		tticks = [ (vMax, str(vMax)), (vMin, str(vMin)) ]
		if self.rs.thresholdValue is not None:
			tticks.append((self.rs.thresholdValue, str(self.rs.thresholdValue) + ' (threshold)'))
		if vMin < 0. and vMax > 0.:
			tticks.append((0.0, str(0.0)))
		tticks = sorted(tticks, key = lambda p: p[0])
		for v, n in tticks:
			ly = V2Y(v)
			ax.add_line(Line2D([ coordEnd + 0. * 0.006 * width, coordEnd + 2. * 0.006 * width ],
				[ ly, ly ],
				color = (0.2, 0.2, 0.2, 1.),
				linewidth = 1.,
				clip_on = False))
			plt.text(coordEnd + 3. * 0.006 * width,
					ly,
					n,
					verticalalignment = 'center',
					horizontalalignment = 'left',
					color = ax.yaxis.label.get_color(),
					fontsize = 8,
					clip_on = False)
		# Thresholded
		if self.rs.thresholdValue is not None:
			tthr = V2Y(self.rs.thresholdValue)
			ax.plot(
				[ coordStart, coordEnd ],
				[ tthr, tthr ],
				linestyle = '-',
				color = 'grey',
				label = 'Expected at random')
			frs = self.rs.regions().filter('', lambda r: r.seq == chromosome\
				and r.end >= coordStart\
				and r.start <= coordEnd)
			for r in frs:
				drawRoundBox(ax = ax, start = r.start, end = r.end, width = width,
					y = self.yB - predHeight + rmargin, height = predHeight - 2*rmargin,
					maxYB = maxYB,
					fc = colour[:3] + [0.4],
					ec = [ c*0.5 for c in colour[:3] ] + [1.])
			# Legend
			geneheight = 1. - 2.*rmargin
			drawRoundBox(ax = ax, start = coordEnd + width*0.01, width = width,
				end = coordEnd + width*0.01 + 3. * 0.006 * width,
				y = self.yB - 0.1 - geneheight, maxYB = maxYB,
				height = geneheight,
				fc = colour[:3] + [0.4],
				ec = [ c*0.5 for c in colour[:3] ] + [1.],
				rlabel = 'Thresholded',
				clip_on = False)

class GenomePlotTrackCVPredictions:
	def __init__(self, rs):
		self.rs = rs
	def getHeight(self):
		# TODO Add multiclass support
		bpred = self.rs.predictionCurves[positive][0]
		if bpred.thresholdValue is not None:
			return 3.0 + (1. if self.rs.hasCores else 0.)
		else:
			return 1.0
	def getName(self):
		return self.rs.name
	def draw(self, ax, chromosome, coordStart, coordEnd, width, rmargin, maxYB, colour):
		import matplotlib.pyplot as plt
		from matplotlib.patches import Rectangle, Polygon
		from matplotlib.lines import Line2D
		colInvisible = [ 0., 0., 0., 0. ]
		height = self.yB - self.yA
		# TODO Add multiclass support
		bpreds = self.rs.predictionCurves[positive]
		bpred = bpreds[0]
		ccurves = [
			next((c for c in p if c.seq == chromosome), None)
			for p in bpreds
		]
		ccurve = ccurves[0]
		if ccurve == None: return
		# Rasterize - logic required depends on curve type
		if not isinstance(ccurve, fixedStepCurve):
			raise Exception('Error: Prediction curve should be of type fixedStepCurve')
		iA = max(int((coordStart - ccurve.span) / ccurve.step), 0)
		iB = max(
				min(int((coordEnd) / ccurve.step), len(ccurve)-1),
			0)
		span = ccurve.span
		step = ccurve.step
		#
		res = 2048
		srcXA = iA * ccurve.step #coordStart
		srcXB = iB * ccurve.step #coordEnd
		scale = res / (srcXB - srcXA)
		ret = [ None for _ in range(res) ]
		for i in range(1+iB-iA):
			vals = [ c[i+iA] for c in ccurves ]
			vMean = mean(vals)
			vCI = CI(vals)
			dstXA = int(min(max(
				i * ccurve.step * res / ((iB - iA) * ccurve.step),
				0),
				res-1))
			dstXB = int(min(max(
				res * (i + ccurve.span / ccurve.step) / (iB - iA),
				0),
				res-1))
			for x in range(dstXA, dstXB+1):
				Q = (ccurve.step * (iA + (iB - iA) * float(x) / res), vMean, vCI)
				if ret[x] is None or Q[1] > ret[x][1]:
					ret[x] = Q
		raster = [ v for v in ret if v is not None ]
		# Plot curve
		vBase = 0.
		vMin = min(y for x, y, ci in raster)
		vMax = max(y for x, y, ci in raster)
		yBottom = min(vBase, vMin)
		yTop = vMax
		cheight = height
		predHeight = 1.
		def V2Y(v):
			return self.yA + rmargin + ((v + vBase - yBottom) * vScale)
		if bpred.thresholdValue is not None:
			yBottom = min(yBottom, bpred.thresholdValue)
			yTop = max(yTop, bpred.thresholdValue)
			cheight -= predHeight
			if self.rs.hasCores:
				cheight -= predHeight
		vScale = (cheight - rmargin*2.) / (yTop - yBottom)
		
		polypts = [ ]
		polypts += [
			(x, V2Y(y+ci))
			for x, y, ci in raster ]
		polypts += reversed([
			(x, V2Y(y-ci))
			for x, y, ci in raster ])
		poly = Polygon(
			polypts,
			fc = colour[:3] + [0.3],
			ec = colInvisible)
		ax.add_patch(poly)
		
		polypts = [ ]
		polypts += [
			(x, V2Y(max(y, vBase)))
			for x, y, ci in raster ]
		polypts += reversed([
			(x, V2Y(min(y, vBase)))
			for x, y, ci in raster ])
		poly = Polygon(
			polypts,
			fc = colour[:3] + [0.5],
			ec = colInvisible)
		
		ax.add_patch(poly)
		ax.add_line(Line2D([ x for x, y, ci in raster ],
			[ V2Y(y) for x, y, ci in raster ],
			linewidth = 0.8,
			color = [ c*0.75 for c in colour[:3] ] + [0.75]))
		# Legend
		lyA = V2Y(vMin)
		lyB = V2Y(vMax)
		ax.add_patch(Rectangle(
			(coordEnd, lyA),
			1. * 0.006 * width,
			lyB-lyA,
			fc = (0.2, 0.2, 0.2, 1.),
			ec = colInvisible,
			clip_on = False
		))
		tticks = [ (vMax, str(vMax)), (vMin, str(vMin)) ]
		thrMean = None
		if bpred.thresholdValue is not None:
			thr = [ p.thresholdValue for p in bpreds ]
			thrMean = mean(thr)
			thrCI = CI(thr)
			tticks.append((thrMean, '%.2f +/- %.2f'%(thrMean, thrCI) + ' (threshold)'))
		if vMin < 0. and vMax > 0.:
			tticks.append((0.0, str(0.0)))
		tticks = sorted(tticks, key = lambda p: p[0])
		for v, n in tticks:
			ly = V2Y(v)
			ax.add_line(Line2D([ coordEnd + 0. * 0.006 * width, coordEnd + 2. * 0.006 * width ],
				[ ly, ly ],
				color = (0.2, 0.2, 0.2, 1.),
				linewidth = 1.,
				clip_on = False))
			plt.text(coordEnd + 3. * 0.006 * width,
					ly,
					n,
					verticalalignment = 'center',
					horizontalalignment = 'left',
					color = ax.yaxis.label.get_color(),
					fontsize = 8,
					clip_on = False)
		# Thresholded
		if bpred.thresholdValue is not None:
			tthr = V2Y(thrMean)
			tthrCI = V2Y(thrMean + thrCI) - tthr
			ax.add_patch(Rectangle(
				(coordStart, tthr - tthrCI),
				coordEnd - coordStart,
				tthrCI * 2.,
				fc = (0.4, 0.4, 0.4, 0.3),
				ec = colInvisible,
				clip_on = False
			))
			ax.plot(
				[ coordStart, coordEnd ],
				[ tthr, tthr ],
				linestyle = '-',
				color = 'grey',
				label = 'Expected at random')
			for rs in self.rs.regions(label = positive):
				frs = rs.filter('', lambda r: r.seq == chromosome\
					and r.end >= coordStart\
					and r.start <= coordEnd)
				for r in frs:
					drawRoundBox(ax = ax, start = r.start, end = r.end, width = width,
						y = self.yB - predHeight + rmargin, height = predHeight - 2*rmargin,
						maxYB = maxYB,
						fc = colour[:3] + [0.4 / len(bpreds)],
						ec = [ c*0.5 for c in colour[:3] ] + [1. / len(bpreds)])
			if self.rs.hasCores:
				for c in self.rs.predictionCurves[positive]:
					frs = c.regions().filter('', lambda r: r.seq == chromosome\
						and r.end >= coordStart\
						and r.start <= coordEnd)
					for r in frs:
						drawRoundBox(ax = ax, start = r.start, end = r.end, width = width,
							y = self.yB - predHeight*2 + rmargin, height = predHeight - 2*rmargin,
							maxYB = maxYB,
							fc = colour[:3] + [0.4 / len(bpreds)],
							ec = [ c*0.5 for c in colour[:3] ] + [1. / len(bpreds)])
			# Legend
			geneheight = 1. - 2.*rmargin
			drawRoundBox(ax = ax, start = coordEnd + width*0.01, width = width,
				end = coordEnd + width*0.01 + 3. * 0.006 * width,
				y = self.yB - 0.1 - geneheight, maxYB = maxYB,
				height = geneheight,
				fc = colour[:3] + [0.4],
				ec = [ c*0.5 for c in colour[:3] ] + [1.],
				rlabel = 'Cores' if self.rs.hasCores else 'Thresholded',
				clip_on = False)
			if self.rs.hasCores:
				drawRoundBox(ax = ax, start = coordEnd + width*0.01, width = width,
					end = coordEnd + width*0.01 + 3. * 0.006 * width,
					y = self.yB - 0.1*2 - geneheight*2, maxYB = maxYB,
					height = geneheight,
					fc = colour[:3] + [0.4],
					ec = [ c*0.5 for c in colour[:3] ] + [1.],
					rlabel = 'Thresholded',
					clip_on = False)

#------------------------------------

def plotGenomeTracks(tracks, chromosome, coordStart, coordEnd, style = 'ggplot', outpath = None, figsize = None):
	if figsize is None:
		figsize = (13., 1. * len(tracks))
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
		from matplotlib.transforms import ScaledTranslation
		import matplotlib.ticker as mticker
		import struct
		with plt.style.context(style):
			fig, ax = plt.subplots(1, figsize = figsize)
			# Set up axes
			plt.yticks(range(len(tracks)))
			ax.set_xlim(coordStart, coordEnd)
			ax.set_ylim(0, len(tracks))
			ticks = ax.get_xticks().tolist()
			ax.xaxis.set_major_locator(mticker.FixedLocator(ticks))
			ax.set_xticklabels([ coordFmt(t) for t in ticks ], fontsize = 12)
			ax.set_yticklabels([  ])
			setYA = []
			setYB = []
			width = coordEnd - coordStart
			# Create plotting tracks
			y = 0.
			ptracks = []
			for rs in tracks:
				yA = y
				nrs = None
				if isinstance(rs, GenomePlotTrack):
					nrs = rs
				elif isinstance(rs, genome):
					nrs = GenomePlotTrackGenome(rs)
				elif isinstance(rs, curves):
					nrs = GenomePlotTrackCurve(rs)
				elif isinstance(rs, CVModelPredictions):
					nrs = GenomePlotTrackCVPredictions(rs)
				else:
					nrs = GenomePlotTrackRegions(rs)
				y += nrs.getHeight()
				ptracks.append(nrs)
				nrs.yA = yA
				nrs.yB = y
				setYA.append(yA)
				setYB.append(y)
			# Plot names
			for rs in ptracks:
				plt.text(coordStart - width*0.015,
					(rs.yB + rs.yA) / 2. + 0.05,
					rs.getName(),
					verticalalignment = 'top',
					horizontalalignment = 'right',
					color = ax.yaxis.label.get_color(),
					fontsize = 15,
					rotation = 45,
					clip_on = False)
			# Plot ticks and labels
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
			# Draw
			rmargin = 0.2
			for rsi, rs in enumerate(ptracks):
				colourRaw = plt.rcParams['axes.prop_cycle'].by_key()['color'][rsi % len(plt.rcParams['axes.prop_cycle'].by_key()['color'])]
				colour = [ v/255. for v in struct.unpack('BBB', bytes.fromhex(colourRaw[1:])) ]
				rs.draw(ax = ax, chromosome = chromosome, coordStart = coordStart, coordEnd = coordEnd, width = width, rmargin = rmargin,
					maxYB = max(setYB), colour = colour)
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

