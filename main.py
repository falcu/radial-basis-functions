from providers import ExcelDataProvider, AutoInferAdapterProvider, RawDataProvider, FilterDataProvider
from radial_basis_functions import KMeans, RadialBasisFunctions

def main():
    fileName = _filePath('test_data','TestData.xlsx')
    provider = ExcelDataProvider(fileName, 'Raw data')
    adapterProvider = AutoInferAdapterProvider(provider, [1, 2])
    rawDataProvider = RawDataProvider( adapterProvider )

    kMeans = KMeans(rawDataProvider,5)
    return kMeans

def testWeights():
    fileName = _filePath('test_data', 'TestData.xlsx')
    excelProvider = ExcelDataProvider(fileName, 'Raw data')
    adapterProvider = AutoInferAdapterProvider(excelProvider, [1, 2])
    inputProvider = RawDataProvider(adapterProvider)
    outputProvider = RawDataProvider(FilterDataProvider(excelProvider,['output']))
    kMeans = KMeans(inputProvider, 5)
    rad = RadialBasisFunctions( outputProvider, kMeans)

    return rad


def testInput():
    fileName = _filePath('test_data', 'TestData.xlsx')
    provider = ExcelDataProvider(fileName, 'Raw data')
    adapterProvider = AutoInferAdapterProvider(provider, [1, 2])
    rawDataProvider = RawDataProvider(adapterProvider)

    return rawDataProvider.getData()

def testOutput():
    fileName = _filePath('test_data', 'TestData.xlsx')
    excelProvider = ExcelDataProvider(fileName, 'Raw data')
    outputProvider = RawDataProvider(FilterDataProvider(excelProvider,['output']))
    return outputProvider.getData()

def _filePath(*args):
    import os
    mainPath = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(mainPath,*args)

if __name__ == '__main__':
    main()