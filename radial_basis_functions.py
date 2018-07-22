import numpy as np
import matplotlib.pyplot as plt
from decorators import computeBefore

class KMeans:

    def __init__(self, provider, numberOfCentroids):
        self.provider = provider
        self._data = None
        self._numberOfCentroids = numberOfCentroids
        self._centroidsData = None

    @computeBefore
    def computeCentroids(self):
        '''Loyd's Algorithm'''

        self._initializeCentroids()
        for i in range(self.numOfCentroids(),self.numOfInputs()):
            vector = self._data[i]
            closest = self._addVectorToClosestCentroid(vector)
            self._recomputeMu(closest)

    def preCompute(self):
        self._data = self.provider.getData()


    def plot(self, dimension1=0, dimension2=1):
        for i in range(0,self.numOfCentroids()):
            byDimensionData = self._centroidsData[i]['points'].transpose()
            byDimensionCentroid = self._centroidsData[i]['mu'].transpose()
            plt.scatter(byDimensionData[dimension1],byDimensionData[dimension2])
            plt.scatter(byDimensionCentroid[dimension1],byDimensionCentroid[dimension2],
                        color='black', marker='x', s=60)
        plt.show()

    def _initializeCentroids(self):
        self._centroidsData = {i:{'mu':0., 'sigma':0., 'points':np.array([])} for
                               i in range(0,self.numOfCentroids())}
        initialData = self._data[0:self.numOfCentroids()]
        for i in range(0,self.numOfCentroids()):
            self._centroidsData[i]['mu'] = initialData[i]
            self._centroidsData[i]['points'] = self._centroidsData[i]['mu']
            self._centroidsData[i]['sigma'] = 0.0

    def _addVectorToClosestCentroid(self, vector):
        centroidDistancePair = []
        for i in range(0,self.numOfCentroids()):
            centroidDistancePair.append((i,self._euclideanDistance(vector,self._muOfCentroid(i))))

        centroidDistancePair.sort( key = lambda v : v[1])
        closest = centroidDistancePair[0][0]
        self._centroidsData[closest]['points'] = np.vstack([self._centroidsData[closest]['points'], vector])
        return closest

    def _recomputeMu(self, centroidIndex):
        points = self._centroidsData[centroidIndex]['points']
        newMu = np.sum(points,axis=0)/len(points)
        self._centroidsData[centroidIndex]['mu'] = newMu

    def _euclideanDistance(self, vector1, vector2):
        return np.linalg.norm(vector1-vector2)

    def _muOfCentroid(self, i):
        return self._centroidsData[i]['mu']

    def numOfCentroids(self):
        return self._numberOfCentroids

    def numOfInputs(self):
        return self._data.shape[0]

def testKMeans():
    import providers

    file = r'D:\Guido\Master Finanzas\2018\Primer Trimestre\Metodos No Param\TestData.xlsx'
    provider = providers.ExcelDataProvider(file, 'Raw data')
    columns = [1, 2]
    #columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    filteredProvider = providers.FilteredProvider(provider, columns)
    data = providers.AutoInferDataAdapter(filteredProvider).adapt().values

    kMeans = KMeans(data,5)
    kMeans.plot()