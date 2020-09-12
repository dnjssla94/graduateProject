from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class Trackable:	# Object Detection되는 객체들.
	def __init__(self, objectID, centroid):
		self.objectID = objectID	# 객체의 고유번호
		self.centroids = [centroid]	# 객체의 중심점 좌표-> 변화시마다 추가된다
		self.directions = []		# 여러값을 넣고 평균값으로 방향을 정할 것
		self.counted = False		# 노랑 boundary를 지나서 한번 카운트 하면 또 세지 않도록.

class Tracker:	# Object Detection하는 기능.
	def __init__(self, maxDisappeared=10):
		self.nextObjectID = 0
		self.objects = OrderedDict()	# OrderedDict(): 입력된 item의 순서를 기억하는 딕셔너리.
		self.disappeared = OrderedDict()
		self.maxDisappeared = maxDisappeared
		self.positions = []

	def register(self, centroid):	# Trackable객체 생성.
		self.objects[self.nextObjectID] = centroid
		self.disappeared[self.nextObjectID] = 0
		self.nextObjectID += 1

	def deregister(self, objectID):	# Trackable객체 소멸.
		del self.objects[objectID]
		del self.disappeared[objectID]
		print('DEREGISTERED : ' + str(objectID))
                
	def update(self, rects):	# Trackable 객체 모니터랑하며 다양한 기능 수행.
		# 기본적으로 3가지 기능을 수행한다.
		# 1. rets배열안의 좌표값을 이용해서 객체의 중심지점을 계산하여 inputCentroids배열에 저장하기.
		# 2. 등록되지 않은 객체는 register()를 이용하여 등록.
		# 3. 사람이 화면에서 나가면 deregister()를 이용하여 삭제.
		# 또 tracker객체가 생성될때 미리 정해둔 maxDisappear값인 50번 만큼 계속
		# 자리에 머물고 있다면 화면에 없다고 판단하고 객체를 삭제한다.
		
		if len(rects) == 0:
			for objectID in list(self.disappeared.keys()):
				self.disappeared[objectID] += 1
				if self.disappeared[objectID] > self.maxDisappeared:
					self.deregister(objectID)

			return self.objects

		inputCentroids = np.zeros((len(rects), 2), dtype="int")

		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)

		if len(self.objects) == 0:
			for i in range(0, len(inputCentroids)):
				self.register(inputCentroids[i])
		else:
			objectIDs = list(self.objects.keys())
			objectCentroids = list(self.objects.values())

			D = dist.cdist(np.array(objectCentroids), inputCentroids)

			rows = D.min(axis=1).argsort()

			cols = D.argmin(axis=1)[rows]

			usedRows = set()
			usedCols = set()

			for (row, col) in zip(rows, cols):
				if row in usedRows or col in usedCols:
					continue

				objectID = objectIDs[row]
				self.objects[objectID] = inputCentroids[col]
				self.disappeared[objectID] = 0
				
				usedRows.add(row)
				usedCols.add(col)

			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			if D.shape[0] >= D.shape[1]:
				for row in unusedRows:
					objectID = objectIDs[row]
					self.disappeared[objectID] += 1

					if self.disappeared[objectID] > self.maxDisappeared:
						self.deregister(objectID)
			else:
				for col in unusedCols:
					self.register(inputCentroids[col])

		return self.objects
