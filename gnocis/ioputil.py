# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

progressTasks = []

class progressTask:
	
	def __init__(self, desc, steps):
		self.desc = desc
		self.step = 0
		self.steps = steps
		self.lastDisplay = ''
		progressTasks.append(self)
	
	def update(self, step):
		self.step = step
		self.display()
	
	def display(self, pLen = 20):
		xA = 0.
		xB = 1.
		desc = ''
		#
		for t in progressTasks:
			stepSize = (xB - xA) / t.steps
			cxA = xA + stepSize * t.step
			xA, xB = cxA, cxA + stepSize
			desc += ' - %s (%d/%d)'%(t.desc, t.step+1, t.steps)
		#
		pI = int(xB * pLen)
		pL = pLen - pI
		perc = 100. * xB
		bar = '[' + ('|'*pI) + (' '*pL) + '] %5.2f %%'%perc + desc
		self.lastDisplay = bar
		print(bar, end = '\r')
	
	def done(self):
		print(' ' * len(self.lastDisplay), end = '\r')
		global progressTasks
		progressTasks = [ t for t in progressTasks if t != self ]

def _tblFieldConv(v, fpdec = 5, strcroplen = 16):
	if isinstance(v, int):
		return str(v)
	if isinstance(v, float):
		return ('%%.%df'%fpdec)%v
	v = str(v)
	if strcroplen > 0 and len(v) > strcroplen:
		return v[:int(strcroplen/2)-2] + '...' + v[-int(strcroplen/2)+1:]
	return v

class nctable:
	"""
	The `nctable` class represents a data table. This data table implements a selection of table manipulation methods, and supports formatted, cropped output in text or to HTML for IPython. In addition, the table supports conversion to Numpy arrays and Pandas DataFrames, in order to facilitate wider use cases.
	
	:param title: Title of the table.
	:param _dict: Data in a dictionary. Each key corresponds to a table column.
	:param align: Dictionary of alignment options, from the same keys as in _dict to an alignment option string per column. Use 'l' for left-alignment, or 'r' for right-alignment. If not set, all columns default to right alignment.
	:param ntop: Number of rows to show on the top of the table.
	:param nbottom: Number of rows to show on the bottom of the table.
	:param maxhorz: Maximum number of columns to show.
	:param spaces: Number of spaces to add between columns.
	:param fpdec: Number of decimals to include for floating point numbers.
	:param strcroplen: Maximum string length.
	:param indexName: Name of index column.
	
	:type title: str
	:type _dict: dict
	:type align: dict, optional
	:type ntop: int, optional
	:type nbottom: int, optional
	:type maxhorz: int, optional
	:type spaces: int, optional
	:type fpdec: int, optional
	:type strcroplen: int, optional
	:type indexName: str, optional
	"""
	
	def __init__(self, title, _dict, align = None, ntop = 5, nbottom = 5, maxhorz = 9, spaces = 3, fpdec = 4, strcroplen = 24, indexName = '__index'):
		self.title = title
		if isinstance(_dict, list):
			# Enables table construction by supplying a list of dicts instead of a base dict
			# The nesting is to ensure that order is preserved
			table = _dict
			_dict = {  }
			for e in table:
				for k in e:
					if k not in _dict:
						_dict[k] = [ x[k] if k in x else '' for x in table ]
		self.ntop, self.nbottom = ntop, nbottom
		self.maxhorz = maxhorz
		self.spaces = spaces
		self.align = align
		self.indexName = indexName
		self.nrows = max([len(_dict[k]) for k in _dict if k != self.indexName])
		self.fpdec = fpdec
		self.strcroplen = strcroplen
		if indexName not in _dict.keys():
			self._dict = {
				**{
					self.indexName: list(range(self.nrows))
				},
				**_dict
			}
		else:
			self._dict = _dict
	
	def full(self):
		"""
		Returns a clone of the table, with cropping disabled.
		
		:return: Table without cropping
		:rtype: nctable
		"""
		return nctable(self.title, self._dict, align = self.align, ntop = 0, nbottom = 0, maxhorz = 0, spaces = self.spaces, strcroplen = 0, indexName = self.indexName)
	
	def noCropNames(self):
		"""
		Returns a clone of the table, with cropping of text disabled.
		
		:return: Table without cropping of text
		:rtype: nctable
		"""
		return nctable(self.title, self._dict, align = self.align, ntop = self.ntop, nbottom = self.nbottom, maxhorz = self.maxhorz, spaces = self.spaces, strcroplen = 0, indexName = self.indexName)
	
	def sort(self, key, ascending = True):
		"""
		Returns a clone sorded by a column.
		
		:param key: Column name to sort by.
		:param ascending: If True, then the table will be sorted in ascending order. Otherwise, descending.
		
		:type key: str
		:type ascending: bool, optional
		
		:return: Sorted table
		:rtype: nctable
		"""
		order = sorted(
				[
					i for i, v in enumerate(self._dict[key])
					if isinstance(v, float) or isinstance(v, int)
				],
				key = lambda x: self._dict[key][x]
			) + sorted(
				[
					i for i, v in enumerate(self._dict[key])
					if not (isinstance(v, float) or isinstance(v, int))
				],
				key = lambda x: self._dict[key][x]
			)
		if not ascending: order = list(reversed(order))
		_dict = {
			k: [
				self._dict[k][i]
				for i in order
			]
			for k in self._dict
		}
		return nctable(self.title, _dict, align = self.align, ntop = self.ntop, nbottom = self.nbottom, spaces = self.spaces, strcroplen = self.strcroplen, indexName = self.indexName)
	
	def drop(self, fields):
		"""
		Returns a clone with columns dropped.
		
		:param fields: List of columns to drop. Alternatively, a single string with a column name can be given.
		
		:type fields: str, list
		
		:return: Table with columns dropped
		:rtype: nctable
		"""
		if isinstance(fields, str):
			fields = [ fields ]
		_dict = {
			k: self._dict[k]
			for k in self._dict
			if k not in fields
		}
		return nctable(self.title, _dict, align = self.align, ntop = self.ntop, nbottom = self.nbottom, spaces = self.spaces, strcroplen = self.strcroplen, indexName = self.indexName)
	
	def summary(self):
		"""
		Returns a summary table, with means, variances and standard deviations of each field.
		
		:return: Summary table
		:rtype: nctable
		"""
		src = self._dict
		keys = [ k for k in src if k != self.indexName ]
		isnum = {
			k: all(isinstance(x, float) or isinstance(x, int) for x in src[k])
			for k in keys
		}
		mean = {
			k: sum(src[k]) / len(src[k]) if isnum[k] else 'N/A'
			for k in keys
		}
		var = {
			k: sum((v - mean[k])**2. for v in src[k]) if isnum[k] else 'N/A'
			for k in keys
		}
		std = {
			k: ((var[k]/(len(src[k])-1))**0.5 if len(src[k]) > 1 else 0) if isnum[k] else 'N/A'
			for k in keys
		}
		_dict = {
			'Field': [ k for k in keys ],
			'Type': [ 'Number' if isnum[k] else 'Text' for k in keys ],
			'Mean': [ mean[k] for k in keys ],
			'Var.': [ var[k] for k in keys ],
			'St.d.': [ std[k] for k in keys ],
		}
		return nctable('Summary table: ' + self.title, _dict, align = { 'Field': 'l' })
	
	def __iter__(self):
		return self._dict.__iter__()
	
	def __getitem__(self, i):
		return self._dict[i]
	
	def __array__(self):
		try:
			import numpy as np
			values = {}
			types = {}
			for k in self._dict:
				if k == self.indexName: continue
				if all(isinstance(v, float) for v in self._dict[k]):
					values[k] = self._dict[k] + [
						np.na for _ in range(len(self._dict[k]), self.nrows) ]
					types[k] = 'f8'
				elif all(isinstance(v, int) for v in self._dict[k]):
					values[k] = self._dict[k] + [
						np.na for _ in range(len(self._dict[k]), self.nrows) ]
					types[k] = 'i4'
				else:
					values[k] = [ str(v) for v in self._dict[k] ] + [
						'' for _ in range(len(self._dict[k]), self.nrows) ]
					types[k] = 'a%d'%( max( len(v) for v in values[k] ) + 1 )
			return np.array([
					tuple([
						values[k][i]
						for k in self._dict if k != self.indexName
					])
					for i in range(self.nrows)
				],
				dtype=[
					(k, types[k])
					for k in self._dict if k != self.indexName
				])
		except ImportError as err:
			raise err
	
	def to_numpy(self):
		"""
		Returns a Numpy array generated from the table.
		
		:return: Numpy array
		:rtype: numpy.array
		"""
		return self.__array__()
	
	def to_pandas(self):
		"""
		Returns a Pandas DataFrame generated from the table.
		
		:return: Pandas DataFrame
		:rtype: pandas.DataFrame
		"""
		try:
			import pandas as pd
			return pd.DataFrame(self.__array__())
		except ImportError as err:
			raise err
	
	def _cropv(self):
		_dict, ntop, nbottom = self._dict, self.ntop, self.nbottom
		# Horizontal cropping
		keys = [ k for k in _dict ]
		if len(keys) > self.maxhorz and self.maxhorz != 0:
			_dict = {
				**{
					k: _dict[k]
					for k in keys[:int(self.maxhorz/2)]
				},
				**{
					'...': [ '...' for _ in range(max(len(_dict[k]) for k in _dict)) ]
				},
				**{
					k: _dict[k]
					for k in keys[-int(self.maxhorz/2):]
				},
			}
		# Vertical cropping
		if ntop == 0 and nbottom == 0:
			return _dict
		if self.nrows <= ntop + nbottom:
			return _dict
		rdict = { k: [] for k in _dict }
		if ntop > 0:
			for i in range(ntop):
				for k in _dict:
					rdict[k].append(_dict[k][i] if i < len(_dict[k]) else '')
			if nbottom > 0:
				for k in _dict:
					rdict[k].append('...')
		if nbottom > 0:
			for i in range(self.nrows-nbottom, self.nrows):
				for k in _dict:
					rdict[k].append(_dict[k][i] if i < len(_dict[k]) else '')
		return rdict
	
	def _prepv(self):
		_dict = self._cropv()
		return {
			_tblFieldConv(k, strcroplen = self.strcroplen): [
				_tblFieldConv(v, fpdec = self.fpdec, strcroplen = self.strcroplen)
				for v in _dict[k]
			]
			for k in _dict
		}
	
	def __repr__(self):
		sp = ''.join(' ' for _ in range(self.spaces))
		_dict = self._prepv()
		align = {
			k: self.align[k]
			if ((self.align is not None) and k in self.align.keys())
			else 'r'
			for k in _dict
		}
		fmt = {
			k: ('%%-%ds' if align[k] == 'l' else '%%%ds')%max([
				0 if k == self.indexName else len(k)
			] + [
				len(e) for e in _dict[k]
			])
			for k in _dict
		}
		hdr = sp.join(
			fmt[k]%('' if k == self.indexName else k)
			for k in _dict
		)
		nrows = max([len(_dict[k]) for k in _dict if k != self.indexName])
		body = '\n'.join(
			sp.join(
				fmt[k]%(_dict[k][i] if i < len(_dict[k]) else '')
				for k in _dict
			)
			for i in range(nrows)
		)
		return self.title + '\n' +\
			'Rows: %d\n'%self.nrows +\
			'Columns: %d\n'%len([ 1 for k in self._dict if k != self.indexName ]) +\
			hdr + '\n' + body
	
	def _repr_html_(self):
		_dict = self._prepv()
		sec = lambda x: x.replace('<', '&lt;').replace('>', '&gt;')
		hdr = '<thead><th></th><th>' + '</th><th>'.join(
			'<b>%s</b>'%str(sec(k))
			for k in _dict if k != self.indexName
		) + '</th></thead>'
		nrows = max([len(_dict[k]) for k in _dict if k != self.indexName])
		body = '<tbody><tr>' + '</tr><tr>'.join(
			'<td><b>%s</b></td>'%_dict[self.indexName][i] + '<td>' + '</td><td>'.join(
				sec(_dict[k][i]) if i < len(_dict[k]) else ''
				for k in _dict if k != self.indexName
			) + '</td>'
			for i in range(nrows)
		) + '</tr></tbody>'
		return '<div><div>' + sec(self.title) + '</div>' +\
			'<div>Rows: %d</div>'%self.nrows +\
			'<div>Columns: %d</div>'%len([ 1 for k in self._dict if k != self.indexName ]) +\
			'<table>' +\
			hdr + body + '</table></div>'

