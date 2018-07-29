import numpy as np
import matplotlib.pyplot as plt
from decorators import computeBefore

class Point:
    def __init__(self, value, id=0):
        self.id = id
        self._value = value

    def value(self):
        return self._value

    def setValue(self, value):
        self._value = value

    def euclideanDistance(self, otherPoint):
        return np.linalg.norm(self.value()-otherPoint.value())

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

class PointsMaker():
    def __init__(self):
        self.id = 1

    def makePoint(self, value):
        p = Point(value, id=self.id)
        self.id += 1
        return p

class RadialBasisFunctions:
    def __init__(self, outputProvider, kMeans, basisFunc=None, outputFunc=None):
        self._kMeans = kMeans
        self.outputProvider = outputProvider
        self._basisFunction = basisFunc or normalRadialBasisFunction
        self._outputFunc = outputFunc or oneOutputFunction
        self._weights = None

    def preCompute(self):
        self._outputData = self.outputProvider.getData()

    @computeBefore
    def computeWeights(self,tries=30):
        for i in range(0,tries):
            self._kMeans.computeCentroids()
            euclideanDistMatrix = self._kMeans.euclideanMatrixRaw()
            fi = self._basisFunction(euclideanDistMatrix, self._kMeans.sigmaRaw())
            fiTrans = fi.transpose()
            try:
                fiTransTimesFiInv = np.linalg.inv( np.matmul(fiTrans,fi) )
                self._weights = np.matmul(np.matmul(fiTransTimesFiInv, fiTrans), self._outputData)
                break
            except np.linalg.LinAlgError:
                print("Singular matrix computed on try {}",format(i+1))

        if i == tries:
            raise Exception("Singular Matrix")


    def h(self, pointsProvider):
        points = pointsProvider.getData()
        edm = self._kMeans.computeEuclideanDistMatrix( points )
        edmBasisFunc = self._basisFunction(edm, self._kMeans.sigmaRaw())
        weightsTrans = self._weights.transpose()
        result = edmBasisFunc*weightsTrans
        return self._outputFunc( np.sum(result,axis=1) )

    def error(self, input, output):
        forcastedOutput = self.h(input)
        misses = 0
        outputData = output.getData()
        for i in range(0,len(forcastedOutput)):
            if not (forcastedOutput[i] == outputData[i] ):
                misses+=1

        return misses, len(forcastedOutput)


def normalRadialBasisFunction(euclideanMatrix, sigmas):
    def _func(value):
        return np.exp(-value)
    m = (euclideanMatrix**2)/(2*(sigmas**2))
    return np.apply_along_axis(_func, 0, m )

def oneOutputFunction(output):
    def _binaryFunc( value ):
        return np.where(value > 0.5, 1, 0)
    return np.apply_along_axis(_binaryFunc,0, output)

class KMeans:

    def __init__(self, inputProvider, numberOfCentroids):
        self.inputProvider = inputProvider
        self._data = None
        self._numberOfCentroids = numberOfCentroids
        self._centroidsData = None
        self._pointsMaker = PointsMaker()
        self._points = []

    def plot(self, dimension1=0, dimension2=1):
        for i in range(0,self.numOfCentroids()):
            points = self._getPointsOfCentroidRaw(i)
            mu = self._getMuOfCentroidRaw(i)
            byDimensionData = points.transpose()
            byDimensionCentroid = mu.transpose()
            plt.scatter(byDimensionData[dimension1],byDimensionData[dimension2])
            plt.scatter(byDimensionCentroid[dimension1],byDimensionCentroid[dimension2],
                        color='black', marker='x', s=60)
        plt.show()

    def preCompute(self):
        self._data = self.inputProvider.getData()

    @computeBefore
    def computeCentroids(self, maxError=0.01, maxIter=10, maxAttempts=10):
        '''Loyd's Algorithm'''
        i = 0
        while True:
            self._computeCentroidsImp(maxError=0.01, maxIter=10)
            if not self._shouldRecomputeCentroids() or i>= maxIter:
                break
            print("Attempting to recompute centroids {} of {}".format((i+1),maxAttempts))
            i+=1

    def _shouldRecomputeCentroids(self):
        '''Returns True if centroids should be recomputed'''
        return any(self._getSizeOfCentroid(i) ==1 for i in range(0,self.numOfCentroids()))


    def _computeCentroidsImp(self, maxError=0.01, maxIter=10):
        self._buildPoints()
        latestCentroids = np.zeros((self.numOfCentroids(),self.numOfDimensions()))
        self._initializeCentroids()
        iter = 0
        hasNotConvergedYet = True
        while iter<maxIter and hasNotConvergedYet:
            for i in range(0,self.numOfInputs()):
                aPoint = self._points[i]
                centroidsChanged = self._addVectorToClosestCentroid(aPoint)
                self._recomputeMu(centroidsChanged)
            if self._centroidsConverged(latestCentroids,self.muRaw(),maxError):
                hasNotConvergedYet = False
            latestCentroids = self.muRaw().copy()
            iter+=1

        self._setEuclideanMatrix()
        self._setCentroidSigmas()

    @computeBefore
    def _buildPoints(self):
        self._points = np.array([self._pointsMaker.makePoint(self._data[i]) for i in range(0,self.numOfInputs())])

    def _centroidsConverged(self, centroids1, centroids2, maxError):
        distances = [Point(centroids1[i]).euclideanDistance(Point(centroids2[i])) for i in range(0,self.numOfCentroids())]
        return all(d<=maxError for d in distances )

    def _initializeCentroids(self, points=None):
        points = points or self._initialRandomPoints()
        self._centroidsData = {}
        self._centroidsData['centroids']={i:{'mu':0., 'sigma':0., 'points':np.array([])} for
                               i in range(0,self.numOfCentroids())}
        self._centroidsData['euclidean_matrix']=None
        for i in range(0,self.numOfCentroids()):
            self._setOnCentroid(i,'mu', Point(points[i].value(), id='mu{}'.format(i)))
            self._setOnCentroid(i,'points', np.array([points[i]]))
            self._setOnCentroid(i,'sigma', 0.0)

    def _initialRandomPoints(self):
        indexPoints = np.arange(0,self.numOfInputs())
        np.random.shuffle(indexPoints)
        return self._points.take(indexPoints[0:self.numOfCentroids()])

    def _addVectorToClosestCentroid(self, aPoint):
        closest = self._findClosestCentroid(aPoint)
        centroidsChanged = []
        previousClosest = self._findCentroidOfPoint(aPoint)
        if previousClosest==closest:
            return []
        if closest!=previousClosest:
            #There is a closest centroid for the point
            self._addPointToCentroid(aPoint, closest)
            centroidsChanged.append(closest)
        if previousClosest is not None:
            self._removePointOfCentroid(aPoint, previousClosest)
            centroidsChanged.append(previousClosest)

        return centroidsChanged

    def _findClosestCentroid(self, aPoint):
        centroidDistancePair = []
        for i in range(0, self.numOfCentroids()):
            mu = self._getMuOfCentroid(i)
            centroidDistancePair.append((i, aPoint.euclideanDistance(mu)))

        centroidDistancePair.sort(key=lambda v: v[1])
        closest = centroidDistancePair[0][0]
        return closest

    def _findCentroidOfPoint(self, aPoint):
        for centroid in range(0,self.numOfCentroids()):
            if self._doPointBelongsToCentroid(aPoint, centroid):
                return centroid

        return None

    def _recomputeMu(self, centroids):
        for i in centroids:
            pointsRaw = self._getPointsOfCentroidRaw(i)
            if pointsRaw.size == 0:
                newInitial = self._findNotAssignedPoint()
                self._addPointToCentroid(newInitial,i)
                pointsRaw = self._getPointsOfCentroidRaw(i)
            newMu = np.sum(pointsRaw,axis=0)/len(pointsRaw)
            self._setMuRaw(i,newMu)

    def _findNotAssignedPoint(self):
        found = False
        while not found:
            randomPoint = self._points[np.random.random_integers(self.numOfInputs())-1]
            if not self._findCentroidOfPoint(randomPoint):
                return randomPoint

    def _setEuclideanMatrix(self):
        edm = self.computeEuclideanDistMatrix( self._data )
        self._setCentroidData('euclidean_matrix', edm )

    def _setCentroidSigmas(self):
        # Matrix indexed by centroid
        for i in range(0,self.numOfCentroids()):
            points = self._getPointsOfCentroidRaw(i)
            euclideanDistances = self._computeEDOfPointsToCentroid(i,points)
            variance = np.sum( euclideanDistances**2 )/len(euclideanDistances)
            self._setOnCentroid(i,'sigma',np.sqrt(variance))

    def computeEuclideanDistMatrix(self, points):
        muMatrix = self.muRaw()
        edm = np.zeros((muMatrix.shape[0], points.shape[0]))
        for i in range(0, self.numOfCentroids()):
            edm[i] = self._computeEDOfPointsToCentroid(i,points)

        return edm.transpose()

    def _computeEDOfPointsToCentroid(self, centroid, points):
        mu = self._getMuOfCentroidRaw(centroid)
        return np.sqrt(np.sum((points - mu) ** 2, axis=1))

    def _getOnCentroid(self,centroidIndex,attribute):
        return self._centroidsData['centroids'][centroidIndex][attribute]

    def _getCentroidData(self, attribute):
        return self._centroidsData[attribute]

    def _getMuOfCentroid(self, i):
        return self._getOnCentroid(i,'mu')

    def _getMuOfCentroidRaw(self, i ):
        return self._getOnCentroid(i, 'mu').value()

    def _getMuCentroids(self):
        return [self._getMuCentroids(i) for i in range(0,self.numOfCentroids())]

    def _getSigmaOfCentroid(self, i):
        return self._getOnCentroid(i, 'sigma')

    def _getPointsOfCentroid(self, centroid):
        return self._getOnCentroid(centroid, 'points')

    def _getSizeOfCentroid(self, centroid):
        return len(self._getPointsOfCentroid(centroid))

    def _getPointsOfCentroidRaw(self, centroid):
        return np.array([p.value() for p in self._getPointsOfCentroid(centroid)])

    def _setCentroidData(self, attribute, value):
        self._centroidsData[attribute]=value

    def _setOnCentroid(self,centroidIndex,attribute, value):
        self._centroidsData['centroids'][centroidIndex][attribute]=value

    def _setPointsOfCentroid(self, centroid, points):
        self._setOnCentroid(centroid,'points',points)

    def _setMuRaw(self, centroid, rawData):
        mu = self._centroidsData['centroids'][centroid]['mu']
        mu.setValue( rawData )

    def _doPointBelongsToCentroid(self, aPoint, centroid):
        return aPoint in self._getPointsOfCentroid(centroid)

    def _removePointOfCentroid(self, aPoint, centroid):
        points = self._getPointsOfCentroid(centroid)
        pos = np.where(points==aPoint)[0]
        if pos.size>0:
            newPoints = np.delete(points,pos)
            self._setPointsOfCentroid(centroid, newPoints)
            return True
        else:
            return False

    def _addPointToCentroid(self, aPoint, centroid):
        points = self._getPointsOfCentroid(centroid)
        points = np.append(points, aPoint)
        self._setPointsOfCentroid(centroid, points)

    def numOfCentroids(self):
        return self._numberOfCentroids

    def numOfInputs(self):
        return self._data.shape[0]

    def numOfDimensions(self):
        return self._data.shape[1]

    def euclideanMatrixRaw(self):
        return self._getCentroidData('euclidean_matrix')

    def muRaw(self):
        return np.array([self._getMuOfCentroid(i).value() for i in range(0, self.numOfCentroids())])

    def sigmaRaw(self):
        return np.array([self._getSigmaOfCentroid(i) for i in range(0, self.numOfCentroids())])

def testKMeans():
    import providers

    file = r'D:\Guido\Master Finanzas\2018\Primer Trimestre\Metodos No Param\TestData.xlsx'
    provider = providers.ExcelDataProvider(file, 'Raw data')
    columns = [1, 2]
    #columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    filteredProvider = providers.FilteredProvider(provider, columns)
    data = providers.AutoInferDataAdapter(filteredProvider).adapt().values

    kMeans = KMeans(data,5)
    return kMeans