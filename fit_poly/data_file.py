#coding=utf-8


def saveMatrix(data, filename):
	nRow = len(data)
	if nRow <= 0:
		print 'saveMatrix() >> 空数据: nRow<=0'
		return -1
	nCol = len(data[0])
	if nCol <= 0:
		print 'saveMatrix() >> 空数据: nCol<=0'
		return -1
	
	f = open(filename, 'w')
	f.write(str(nRow) + '\t' + str(nCol) + '\r\n')
	for i in xrange(nRow):
		f.write(str(data[i][0]))
		for j in xrange(1, nCol):
			f.write('\t' + str(data[i][j]))
		f.write('\r\n')
	f.close()


def loadMatrix(filename, dataType):
	f = open(filename, 'r')
	
	line = f.readline()
	if not line:
		print "loadMatrix() >> %s文件, 空数据" %(filename)
		return -1, 0, 0, 0
	lines = line.strip().split('\t')
	if len(lines) != 2:
		print "loadMatrix() >> %s文件, 首行格式错误: %s" %(filename, line)
		return -1, 0, 0, 0
	nRow = int(lines[0])
	nCol = int(lines[1])
	
	data = []
	for i in xrange(0, nRow):
		line = f.readline()
		if not line:
			print "loadMatrix() >> %s文件, 行不足: %d<%d" %(filename, i, nRow)
			return -2, data, nRow, nCol
		lines = line.strip().split('\t')
		if len(lines) != nCol:
			print "loadMatrix() >> %s文件, 第%d行,列不足: %d<%d" %(filename, i, len(lines), nCol)
			return -1, data, nRow, nCol
		
		rowData = []
		for j in xrange(0, nCol):
			if dataType == int:
				rowData.append(int(lines[j]))
			else:
				rowData.append(float(lines[j]))
		data.append(rowData)
	return 0, data, nRow, nCol