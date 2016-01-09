#coding=utf-8


import numpy as np


def savePoints(ptx, pty, filename):
	if len(ptx) != len(pty):
		print 'savePoints() >> ptx和pty的行数不一致'
		return -1
	if len(ptx) < 1:
		print 'savePoints() >> 空数据'
		return -1
	nRow = len(ptx)
	nCol = len(ptx[0])
	if nCol < 1:
		print 'savePoints() >> 列数<1'
		return -1
	
	ptx = np.int32(ptx)
	pty = -np.int32(pty)
	
	f = open(filename, 'w')
	f.write(str(nRow) + '\t' + str(nCol) + '\r\n')
	for i in xrange(nRow):
		f.write( str(ptx[i][0]) + '\t' + str(pty[i][0]) )
		for j in xrange(1, nCol):
			f.write( '\t' + str(ptx[i][j]) + '\t' + str(pty[i][j]) )
		f.write('\r\n')
	f.close()


def loadPoints(filename):
	f = open(filename, 'r')
	
	line = f.readline()
	if not line:
		print "loadPoints() >> %s文件, 空数据" %(filename)
		return -1, 0, 0, 0, 0
	lines = line.strip().split('\t')
	if len(lines) != 2:
		print "loadPoints() >> %s文件, 首行格式错误: %s" %(filename, line)
		return -1, 0, 0, 0, 0
	nRow = int(lines[0])
	nCol = int(lines[1])
	ptx = []
	pty = []
	
	for i in xrange(0, nRow):
		line = f.readline()
		if not line:
			print "loadPoints() >> 行不足: %d<%d" %(i, nRow)
			return -2, nRow, nCol, ptx, pty
		lines = line.strip().split('\t')
		if len(lines) != 2*nCol:
			print "loadPoints() >> 第%d行,列不足: %d<%d" %(i, len(lines), 2*nCol)
			return -1, nRow, nCol, ptx, pty
		x = []
		y = []
		for j in xrange(0, nCol):
			x.append(int(lines[2*j]))
			y.append(-int(lines[2*j+1]))
		ptx.append(x)
		pty.append(y)
	
	f.close()
	return 0, nRow, nCol, ptx, pty