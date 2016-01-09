#coding=utf-8


import matplotlib.pyplot as plt

import data_file as dfile
import point_file as ptfile
import util


def rowParabolaCrossPoint():
	ret1, rowCoeff, nRow, _ = dfile.loadMatrix('data/row_coeff.txt', float)
	ret2, colCoeff, nCol, _ = dfile.loadMatrix('data/col_coeff.txt', float)
	if ret1 != 0 or ret2 != 0:
		print 'curveMap() >> load coeff失败'
		return
	
	for i in xrange(nRow):
		for j in xrange(nRow):
			if j != i:
				x1, y1, x2, y2 = util.parabolaCrossPoint(rowCoeff[i], rowCoeff[j])
				plt.plot(x1, y1, 'b.')
				plt.plot(x2, y2, 'b.')
	plt.show()


if __name__ == '__main__':
	rowParabolaCrossPoint()