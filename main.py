from providers import ExcelDataProvider, AutoInferAdapterProvider, RawDataProvider, \
                        FilterDataProvider, ShuffleProvider, SlidesProvider, ManualDataAdapterProvider, ApplyFuncAdapter
from radial_basis_functions import KMeans, RadialBasisFunctions
import itertools
import numpy as np
from timeit import default_timer as timer
from argparse import ArgumentParser, ArgumentTypeError

def main():
    def betweenZeroAndOneFloat(x):
        x = float(x)
        if x < 0.0 or x > 1.0:
            raise ArgumentTypeError("{} not in range [0.0, 1.0]".format (x,))
        return x

    def optionStr(x, options):
        if not x in options:
            raise ArgumentTypeError("Valid options are {}".format(options))
        return x

    kmeansOptions = lambda x : optionStr(x,['kmeans','error','find_dim'])
    fileNameOptions = lambda x : optionStr(x,['credit','test2dim'])

    parser = ArgumentParser(description='Radial Basis Functions')
    parser.add_argument('-o', '--option', help='Program Option: kmeans, error or find_dim', required=True, type=kmeansOptions)
    parser.add_argument('-fo', '--file_option', help='File Name option: credit or test2dim', required=True, type=fileNameOptions)
    parser.add_argument('-c', '--clusters', help='Number of clusters', required=False, type=int, default=5)
    parser.add_argument('-is', '--insample', help='In Sample proportion', required=False, type=betweenZeroAndOneFloat, default=0.5)
    parser.add_argument('-p', '--plot', help='If it should plot. Makes sense for 2 Dim input', required=False, type=bool,default=False)
    parser.add_argument('-mc', '--min_comb', help='Min combination of columns. Relevant for option find_dim', required=False,
                        type=int, default=5)
    parser.add_argument('-ic', '--iter_conmb', help='Number of iterations per combination. Relevant for option find_dim',
                        required=False,type=int, default=5)
    args = vars(parser.parse_args())
    programOption = args['option']
    fileName = args['file_option']
    nOfClusters = args['clusters']
    slides = [args['insample'],(1.0-args['insample'])]
    shouldPlot = args['plot']
    minComb = args['min_comb']
    interComb = args['iter_conmb']
    _, inputDimensions, _, _ = _getTestFileInfo(fileName)
    if minComb>len(inputDimensions):
        raise ArgumentTypeError("Max Min combinations can be {}".format(inputDimensions))

    if programOption == 'kmeans':
        kmeans(nOfClusters, fileName, plot=shouldPlot)
    elif programOption == 'error':
        computeError(nOfClusters, fileName, slides=slides)
    elif programOption == 'find_dim':
        findRelevantDimensions(nOfClusters,fileName, iterationsPerCombination=interComb, minCombination=minComb)

def provideCreditData(fileProvider,inputColumns,outputColumn,slidesPercentage=[1.0], outputFunc=None):
    outputFunc = outputFunc or (lambda v:v)
    randomProvider = ShuffleProvider(fileProvider)
    inputFilter = FilterDataProvider(randomProvider,columns=inputColumns)
    outputFilter = FilterDataProvider(randomProvider,columns=[outputColumn])
    adaptInputProv = AutoInferAdapterProvider(inputFilter)
    outputAdapter = ManualDataAdapterProvider(outputFilter,adapterMapping={outputColumn:ApplyFuncAdapter(outputFunc)})
    inputRaw = RawDataProvider(adaptInputProv)
    outputRaw = RawDataProvider(outputAdapter)
    inputSlides = SlidesProvider(inputRaw, slidesPercentage=slidesPercentage)
    outputSlides = SlidesProvider(outputRaw, slidesPercentage=slidesPercentage)

    return inputSlides,outputSlides

def makeFileProvider(fileName,sheetName='Raw data'):
    return ExcelDataProvider(fileName, sheetName)

def _getTestFileInfo(name):
    if name == 'credit':
        fileName = _filePath('test_data', 'Base_Clientes Alemanes.xlsx')
        inputColumns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        outputColumn = 'Tipo cliente'
        outputFunc = lambda v: v - 1

    elif name == 'test2dim':
        fileName = _filePath('test_data', 'TestData.xlsx')
        inputColumns = [1, 2]
        outputColumn = 'output'
        outputFunc = None
    else:
        raise Exception("Not test file named {}".format(name))
    return makeFileProvider(fileName), inputColumns, outputColumn, outputFunc


def kmeans(nOfClusters = 6, fileName='test2dim', plot=True):
    fileProvider, inputColumns, outputColumn, outputFunc = _getTestFileInfo(fileName)
    inputProv, outputProv = provideCreditData(fileProvider, inputColumns, outputColumn, outputFunc=outputFunc)

    kMeansSAlg = KMeans(inputProv.getData()[0],nOfClusters)
    kMeansSAlg.computeCentroids()
    print("Centroids: ")
    print(kMeansSAlg.muRaw())
    if plot:
        kMeansSAlg.plot()

def testHReal(nOfClusters = 6):

    fileName = _filePath('test_data', 'Base_Clientes Alemanes.xlsx')
    inputColumns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    outputColumn = 'Tipo cliente'
    slides = [0.8,0.2]

    inputProv, outputProv = provideCreditData(makeFileProvider(fileName), inputColumns, outputColumn,slidesPercentage=slides,
                                              outputFunc=lambda v: v - 1)
    kMeans = KMeans(inputProv.getData()[0], nOfClusters)
    rad = RadialBasisFunctions(outputProv.getData()[0], kMeans, outputFunc = lambda v : v)
    rad.computeWeights()
    print("weights computed")

    return rad.h(inputProv.getData()[1])

def computeError(nOfClusters=5, fileName='credit', slides=None):
    fileProvider, inputColumns, outputColumn, outputFunc = _getTestFileInfo(fileName)
    slides = slides or [0.5,0.5]

    inputProv, outputProv = provideCreditData(fileProvider, inputColumns, outputColumn,slidesPercentage=slides,
                                              outputFunc = outputFunc)
    kMeans = KMeans(inputProv.getData()[0], nOfClusters)
    rad = RadialBasisFunctions(outputProv.getData()[0], kMeans)
    rad.computeWeights()
    misses, total = rad.error(inputProv.getData()[1], outputProv.getData()[1])
    print("{} % error out of sample".format(100*misses/total))

def findRelevantDimensions(nOfClusters=5, fileName='credit', iterationsPerCombination=10, minCombination=5, slides=None):
    fileProvider, inputColumns, outputColumn, outputFunc = _getTestFileInfo(fileName)
    slides = slides or [0.5, 0.5]
    allCombinations = []
    for i in range(minCombination,len(inputColumns)+1):
        allCombinations+= list(itertools.combinations(inputColumns, i ))
    result = { comb:[comb,0,0] for comb in allCombinations}
    progress = 0
    totalCombinations = len(allCombinations)*iterationsPerCombination
    percentReport = int(len(allCombinations)*0.001) or 10
    print("Will compute {} iterations".format(totalCombinations))
    print("Total dimensions combinations {}".format(len(allCombinations)))
    print("Report every {} iterations".format(percentReport))
    start = timer()
    for i in range(0, iterationsPerCombination):
        np.random.shuffle(allCombinations)

        for columComb in allCombinations:
            inputProv, outputProv = provideCreditData(fileProvider, list(columComb), outputColumn,
                                                 slidesPercentage=slides, outputFunc = outputFunc)
            kMeans = KMeans(inputProv.getData()[0], nOfClusters)
            rad = RadialBasisFunctions(outputProv.getData()[0], kMeans)
            rad.computeWeights()
            misses, total = rad.error(inputProv.getData()[1], outputProv.getData()[1])
            currentError=(misses/total)
            result[columComb][1]+= currentError
            result[columComb][2] = result[columComb][1]/(i+1)
            progress+=1
            if progress % percentReport == 0:
                end = timer()
                print("Report after {} minutes".format((end-start)/60.0))
                start = timer()
                print('Progress: {}%'.format(100.0*np.round(progress/totalCombinations,3)))
                _showBestCombination(result, "Best combination at this point:")
                print('--------------------------------------------------------------')
        print("Went through all combinations {} of {}".format(i+1,iterationsPerCombination))

    _showBestCombination(result, "Final Result")
    print('-----------------------------------------------------------')
    allResults = list(result.values())
    allResults.sort(key=lambda v: v[2])
    allResults = [r for r in allResults if r[2] > 0]
    print("Summary")
    print("Combination|Error")
    for r in allResults:
        print("{}|{}".format(r[0],r[2]))


def _showBestCombination(result,label=''):
    partialResult = list(result.values())
    partialResult = [r for r in partialResult if r[2]>0]
    partialResult.sort(key=lambda v: v[2])
    error = partialResult[0][2]
    comb = partialResult[0][0]
    print(label)
    print("Error {}".format(error))
    print("Dimensions: {}".format(comb))

def _filePath(*args):
    import os
    mainPath = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(mainPath,*args)

if __name__ == '__main__':
    main()