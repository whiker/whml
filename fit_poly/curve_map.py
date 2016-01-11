#coding=utf-8


import matplotlib.pyplot as plt
import numpy as np

import data_file as dfile
import point_file as ptfile
import tensorflow_util as tfu
import util


def rowcolParabolaCrossPoint():
	ret1, rowCoeff, nRow, _ = dfile.loadMatrix('data/row_coeff.txt', float)
	ret2, colCoeff, nCol, _ = dfile.loadMatrix('data/col_coeff.txt', float)
	if ret1 != 0 or ret2 != 0:
		print 'curveMap() >> load coeff失败'
		return
	
	for i in xrange(nRow):
		for j in xrange(nRow):
			if j != i:
				x1, y1, x2, y2 = util.parabolaCrossPoint(rowCoeff[i], rowCoeff[j])
				color = 'b.'
				if j == nRow/2:
					color = 'r.'
				plt.plot(x1, y1, color)
				plt.plot(x2, y2, color)
	
	for i in xrange(nCol):
		for j in xrange(nCol):
			if j == nCol/2:
				x1, y1, x2, y2 = util.parabolaCrossPoint(colCoeff[i], colCoeff[j])
				if x1 and y1 and x2 and y2:
					x1, y1, x2, y2 = y1, -x1, y2, -x2
				plt.plot(x1, y1, 'g.')
				plt.plot(x2, y2, 'g.')
	plt.show()


def dim2To1(data):
	if len(data) < 2 or len(data[0]) < 2:
		return
	nRow = len(data)
	nCol = len(data[0])
	newd = []
	for i in xrange(nRow):
		for j in xrange(nCol):
			newd.append(data[i][j])
	return np.array(np.float32(newd))


def merge(x, y):
	if len(x) != len(y):
		return
	return np.array(np.float32([ [x[i], y[i]] for i in xrange(len(x)) ]))


def expandToPoly(x, y):
	nPoly = 12
	data = [1]
	for i in xrange(0, nPoly):
		ind = i * (i+1) / 2
		data.append(data[ind]*x)
		for j in xrange(i+1):
			data.append(data[ind]*y)
			ind = ind+1
	return data


def tfnn(tfin, tfout):
	hideUnitNum = [2, 10, 10, 2]
	tfu.nn(tfin, tfout, hideUnitNum, 100, 0.01, 10)


def tfln(tfin, tfout):
	datax = []
	datay1 = []
	datay2 = []
	for i in xrange(0, len(tfin)):
		datax.append( expandToPoly(tfin[i][0], tfin[i][1]) )
		datay1.append(tfout[i][0])
		datay2.append(tfout[i][1])
	datax = np.transpose(np.array(np.float32(datax)))
	datay1 = np.array(np.float32(datay1))
	datay2 = np.array(np.float32(datay2))
	#w1 = tfu.linearRegression(datax, datay1, 1000000, 0.001, 100)
	w2 = tfu.linearRegression(datax, datay2, 100000, 0.001, 100)
	#dfile.saveVector(w1, 'data/w31.txt')
	dfile.saveVector(w2, 'data/w41.txt')
	#print w1
	print w2


def buildMap():
	ret, nRow, nCol, srcx, srcy = ptfile.loadPoints('data/true_pts.txt')
	if ret != 0:
		print 'buildMap() >> 读取数据失败'
		return
	scale = 50
	dstx = [ [j*scale for j in xrange(nCol)] for i in xrange(nRow) ]
	dsty = [ [i*scale for j in xrange(nCol)] for i in xrange(nRow) ]
	
	srcx = dim2To1(srcx)
	srcy = dim2To1(srcy)
	dstx = dim2To1(dstx)
	dsty = dim2To1(dsty)
	
	tfin = merge(dstx, dsty)
	tfout = merge(srcx, srcy)
	tfln(tfin, tfout)


if __name__ == '__main__':
	rowcolParabolaCrossPoint()
	#buildMap()