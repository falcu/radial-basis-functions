import numpy as np
import matplotlib.pyplot as plt
from decorators import computeBefore

class RadialBasisFunctions:
    def __init__(self, outputProvider, kMeans, basisFunction=None):
        self._kMeans = kMeans
        self.outputProvider = outputProvider
        self._basisFunction = basisFunction or normalRadialBasisFunction
        self._weights = None

    def preCompute(self):
        self._outputData = self.outputProvider.getData()

    @computeBefore
    def computeWeights(self):
        self._kMeans.computeCentroids()
        euclideanDistMatrix = self._kMeans.euclideanMatrixRaw()
        sigmas = self._kMeans.sigmaRaw()
        fi = self._basisFunction(euclideanDistMatrix, sigmas)
        fiTrans = fi.transpose()
        fiTransTimesFiInv = np.linalg.inv( np.matmul(fiTrans,fi) )
        self._weights = np.matmul( np.matmul(fiTransTimesFiInv,fiTrans), self._outputData)

    def h(self, points):
        muCentroids = self._kMeans.muRaw()



def normalRadialBasisFunction(euclideanMatrix, sigmas):
    def _func(value):
        return np.exp(-value)
    m = euclideanMatrix/(sigmas*2)
    return np.apply_along_axis(_func, 0, m )

class KMeans:

    def __init__(self, inputProvider, numberOfCentroids):
        self.inputProvider = inputProvider
        self._data = None
        self._numberOfCentroids = numberOfCentroids
        self._centroidsData = None

    def plot(self, dimension1=0, dimension2=1):
        for i in range(0,self.numOfCentroids()):
            points = self._pointsOfCentroid(i)
            mu = self._muOfCentroid(i)
            byDimensionData = points.transpose()
            byDimensionCentroid = mu.transpose()
            plt.scatter(byDimensionData[dimension1],byDimensionData[dimension2])
            plt.scatter(byDimensionCentroid[dimension1],byDimensionCentroid[dimension2],
                        color='black', marker='x', s=60)
        plt.show()

    def preCompute(self):
        self._data = self.inputProvider.getData()

    @computeBefore
    def computeCentroids(self):
        '''Loyd's Algorithm'''

        self._initializeCentroids()
        for i in range(self.numOfCentroids(),self.numOfInputs()):
            vector = self._data[i]
            closest = self._addVectorToClosestCentroid(vector)
            self._recomputeMu(closest)

        self._computeEuclideanMatrix()
        self._computeCentroidSigmas()
        return self._centroidsData

    def _initializeCentroids(self):
        self._centroidsData = {}
        self._centroidsData['centroids']={i:{'mu':0., 'sigma':0., 'points':np.array([])} for
                               i in range(0,self.numOfCentroids())}
        self._centroidsData['euclidean_matrix']=None

        initialData = self._data[0:self.numOfCentroids()]
        for i in range(0,self.numOfCentroids()):
            self._setOnCentroid(i,'mu', initialData[i])
            self._setOnCentroid(i,'points', initialData[i])
            self._setOnCentroid(i,'sigma', 0.0)

    def _addVectorToClosestCentroid(self, vector):
        centroidDistancePair = []
        for i in range(0,self.numOfCentroids()):
            centroidDistancePair.append((i,self._euclideanDistanceVector(vector,self._muOfCentroid(i))))

        centroidDistancePair.sort( key = lambda v : v[1])
        closest = centroidDistancePair[0][0]
        newPoints = np.vstack([self._pointsOfCentroid(closest), vector])
        self._setOnCentroid(closest,'points',newPoints)
        return closest

    def _recomputeMu(self, centroidIndex):
        points = self._pointsOfCentroid(centroidIndex)
        newMu = np.sum(points,axis=0)/len(points)
        self._setOnCentroid(centroidIndex,'mu',newMu)

    def _computeEuclideanMatrix(self):
        nOfCentroids = self.numOfCentroids()
        nOfInputs = self.numOfInputs()
        muData = self.muRaw()
        muMinusPoints = np.zeros((nOfCentroids, nOfInputs))
        for i in range(0, nOfCentroids):
            muMinusPoints[i] = np.sqrt( np.sum((self._data - muData[i]) ** 2, axis=1) )

        self._setCentroidData('euclidean_matrix', muMinusPoints.transpose())

    def _computeCentroidSigmas(self):
        # Matrix indexed by centroid
        euclideanDistanceMatrix = self._getCentroidData('euclidean_matrix').transpose()
        for i in range(0,self.numOfCentroids()):
            variance = np.sum( euclideanDistanceMatrix[i]**2 )/len(euclideanDistanceMatrix[i])
            self._setOnCentroid(i,'sigma',np.sqrt(variance))

    def _euclideanDistanceVector(self, vector1, vector2):
        return np.linalg.norm(vector1-vector2)

    def euclideanDistanceMatrix(self, points, muMatrix):
        pass

    def _muOfCentroid(self, i):
        return self._getOnCentroid(i,'mu')

    def _sigmaOfCentroid(self, i):
        return self._getOnCentroid(i, 'sigma')

    def _pointsOfCentroid(self, i):
        return self._getOnCentroid(i, 'points')

    def _setOnCentroid(self,centroidIndex,attribute, value):
        self._centroidsData['centroids'][centroidIndex][attribute]=value

    def _setCentroidData(self, attribute, value):
        self._centroidsData[attribute]=value

    def _getOnCentroid(self,centroidIndex,attribute):
        return self._centroidsData['centroids'][centroidIndex][attribute]

    def _getCentroidData(self, attribute):
        return self._centroidsData[attribute]

    def numOfCentroids(self):
        return self._numberOfCentroids

    def numOfInputs(self):
        return self._data.shape[0]

    def euclideanMatrixRaw(self):
        return self._getCentroidData('euclidean_matrix')

    def muRaw(self):
        return np.array([self._muOfCentroid(i) for i in range(0,self.numOfCentroids())])

    def sigmaRaw(self):
        return np.array([self._sigmaOfCentroid(i) for i in range(0, self.numOfCentroids())])

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