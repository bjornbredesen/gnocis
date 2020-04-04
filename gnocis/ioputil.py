# -*- coding: latin-1 -*-
############################################################################
# Gnocis
# Bjørn Bredesen, 2018-2019
# bjorn.bredesen@ii.uib.no
############################################################################

class nctable:
	
	def __init__(self, title, _dict, align = None, ntop = 10, nbottom = 10, spaces = 3):
		self.title = title
		self.ntop, self.nbottom = ntop, nbottom
		self.spaces = spaces
		self.align = align
		self.indexName = '__index'
		self.nrows = max([len(_dict[k]) for k in _dict])
		self._dict = {
			**{
				self.indexName: list(range(self.nrows))
			},
			**_dict
		}
	
	def full(self):
		return nctable(self.title, self._dict, align = self.align, ntop = 0, nbottom = 0, spaces = self.spaces)
	
	def sort(self, key):
		order = [
			x[1]
			for x in sorted(
				zip(self._dict[key], list(range(self.nrows))),
				key = lambda x: x[0]
			)
		]
		_dict = {
			k: [ self._dict[k][i] for i in order ] for k in self._dict
		}
		return nctable(self.title, _dict, align = self.align, ntop = self.ntop, nbottom = self.nbottom, spaces = self.spaces)
	
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
		return self.__array__()
	
	def to_pandas(self):
		try:
			import pandas as pd
			return pd.DataFrame(self.__array__())
		except ImportError as err:
			raise err
	
	def _as_dict_(self):
		return self._dict
	
	def _crop(self):
		_dict, ntop, nbottom = self._dict, self.ntop, self.nbottom
		if ntop == 0 and nbottom == 0:
			return _dict
		self.nrows = max([len(_dict[k]) for k in _dict])
		if self.nrows <= ntop + nbottom:
			return _dict
		rdict = { k: [] for k in _dict }
		if ntop > 0:
			for i in range(ntop):
				for k in _dict:
					rdict[k].append(str(_dict[k][i]) if i < len(_dict[k]) else '')
			if nbottom > 0:
				for k in _dict:
					rdict[k].append('...')
		if nbottom > 0:
			for i in range(self.nrows-nbottom, self.nrows):
				for k in _dict:
					rdict[k].append(str(_dict[k][i]) if i < len(_dict[k]) else '')
		return rdict
	
	def __repr__(self):
		sp = ''.join(' ' for _ in range(self.spaces))
		_dict = self._crop()
		align = {
			k: self.align[k]
			if ((self.align is not None) and k in self.align.keys())
			else 'r'
			for k in _dict
		}
		self.nrows = max([len(_dict[k]) for k in _dict])
		fmt = {
			k: ('%%-%ds' if align[k] == 'l' else '%%%ds')%max([
				0 if k == self.indexName else len(k)
			] + [
				len(str(e)) for e in _dict[k]
			])
			for k in _dict
		}
		hdr = sp.join(
			fmt[k]%('' if k == self.indexName else str(k))
			for k in _dict
		)
		body = '\n'.join(
			sp.join(
				fmt[k]%(str(_dict[k][i]) if i < len(_dict[k]) else '')
				for k in _dict
			)
			for i in range(self.nrows)
		)
		return self.title + '\n' + hdr + '\n' + body
	
	def _repr_html_(self):
		_dict = self._crop()
		self.nrows = max([len(_dict[k]) for k in _dict])
		hdr = '<thead><th></th><th>' + '</th><th>'.join(
			'<b>%s</b>'%str(k)
			for k in _dict.keys() if k != self.indexName
		) + '</th></thead>'
		body = '<tbody><tr>' + '</tr><tr>'.join(
			'<td><b>%s</b></td>'%_dict[self.indexName][i] + '<td>' + '</td><td>'.join(
				str(_dict[k][i]) if i < len(_dict[k]) else ''
				for k in _dict.keys() if k != self.indexName
			) + '</td>'
			for i in range(self.nrows)
		) + '</tr></tbody>'
		return '<div><div>' + self.title + '</div><table>' + hdr + body + '</table></div>'

