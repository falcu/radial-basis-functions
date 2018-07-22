from providers import ExcelDataProvider, AutoInferAdapterProvider, RawDataProvider
from radial_basis_functions import KMeans

def main():
    fileName = _filePath('test_data','TestData.xlsx')
    provider = ExcelDataProvider(fileName, 'Raw data')
    adapterProvider = AutoInferAdapterProvider(provider, [1, 2])
    rawDataProvider = RawDataProvider( adapterProvider )

    kMeans = KMeans(rawDataProvider,5)
    kMeans.computeCentroids()
    kMeans.plot()

def _filePath(*args):
    import os
    mainPath = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(mainPath,*args)

if __name__ == '__main__':
    main()