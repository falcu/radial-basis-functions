import pandas as pd
from decorators import computeBefore
import numpy as np
from abc import ABC, abstractmethod


class Provider(ABC):
    def __init__(self):
        self._data = None
        self._loaded = False

    def preCompute(self):
        if not self._loaded:
            self._data = self.preComputeImp()
            self._loaded = True

    @abstractmethod
    def preComputeImp(self):
        pass

    @computeBefore
    def getData(self):
        return self._data


class NoActionProvider(Provider):
    def __init__(self,data):
        super().__init__()
        self._noActionData = data

    def preComputeImp(self):
        return self._noActionData


class ExcelDataProvider(Provider):
    def __init__(self, fileName, sheetName):
        super().__init__()
        self.fileName = fileName
        self.sheetName = sheetName

    def preComputeImp(self):
        print("Loading {}".format(self.fileName))
        return pd.read_excel(self.fileName, self.sheetName)
        print("File loaded")

class FilterDataProvider(Provider):
    def __init__(self, provider, columns=None,rows=None):
        super().__init__()
        self.provider = provider
        self.columns = columns
        self.rows = rows

    def preComputeImp(self):
        data = self.provider.getData()
        data = data[self.columns] if self.columns else data
        return data.loc[self.rows] if self.rows else data


class AutoInferAdapterProvider(Provider):
    def __init__(self, provider):
        super().__init__()
        self.provider = provider

    def preComputeImp(self):
        return self._adaptData()

    def _adaptData(self):
        data = self.provider.getData()
        convertedData = []
        for col in data.columns.values:
            series = data[col]
            adapter = self._findAdapter(series)
            convertedData.append(adapter.adapt(series))

        return pd.concat(convertedData, axis=1)

    def _findAdapter(self, series):
        rawData = series.values
        if any(isinstance(v,str) for v in rawData):
            return QualitativeToBinary()
        else:
            try:
                [float(v) for v in rawData]
                allValues = list(set(rawData))
                allValues.sort()
                if allValues == [0,1]:
                    return NoConversionAdapter()
                else:
                    return NumericNormalizedAdapter()
            except:
                return QualitativeToBinary()


class RawDataProvider(Provider):
    def __init__(self, provider ):
        super().__init__()
        self.provider = provider

    def preComputeImp(self):
        return self.provider.getData().values


class ShuffleProvider(Provider):
    def __init__(self, provider ):
        super().__init__()
        self.provider = provider

    def preComputeImp(self):
        return self.provider.getData().sample(frac=1)\
                            .reset_index(drop=True)


class SlidesProvider(Provider):
    def __init__(self, provider, slidesPercentage=[1.0]):
        super().__init__()
        if not sum(slidesPercentage)==1.0:
            raise Exception("Slides percentage should add 1.0")
        self.provider = provider
        self._slidesPercentage = slidesPercentage


    def preComputeImp(self):
        data = self.provider.getData()
        inputSize = data.shape[0]
        slides = []
        fromSlidePer = 0.0
        for aSlidePercentage in self._slidesPercentage:
            toSlidePer = fromSlidePer + aSlidePercentage
            fromSlide = int(fromSlidePer*inputSize)
            toSlide = int(toSlidePer*inputSize)
            slides.append( NoActionProvider( data[fromSlide:toSlide] ) )
            fromSlidePer = toSlidePer

        return slides


# Adapters

class ManualDataAdapterProvider(Provider):
    def __init__(self, provider, adapterMapping):
        super().__init__()
        self.provider = provider
        self.adapterMapping = adapterMapping

    def preComputeImp(self):
        def _adapt(file, col):
            adapter = self.adapterMapping.get(col, EmptyAdapter())
            return adapter.adapt(file[col])

        file = self.provider.getData()
        convertedData = [_adapt(file, col) for col in file.columns.values]

        return pd.concat(convertedData, axis=1)


class VectorAdapter(ABC):
    @abstractmethod
    def adapt(self, series):
        pass


class NoConversionAdapter(VectorAdapter):
    def adapt(self, series):
        return pd.DataFrame(series)


class QualitativeToBinary(VectorAdapter):
    def adapt(self, series):
        distintValues = list(set(series.values))
        distintValues.pop() #Remove one of the dummies to avoid multi colinearity
        distinctValuesLen = len(distintValues)
        rawData = np.zeros((distinctValuesLen, len(series.values)))
        for i in range(0, distinctValuesLen):
            rawData[i][np.where(series.values == distintValues[i])] = 1

        colNames = ['{}_{}'.format(series.name, value) for value in distintValues]

        return pd.DataFrame(rawData.transpose(), columns=colNames)


class EmptyAdapter(VectorAdapter):
    def adapt(self, series):
        return pd.DataFrame()


class NumericNormalizedAdapter(VectorAdapter):
    def adapt(self, series):
        mean = series.mean()
        std = series.std()
        normalized = series.apply(lambda v: (v - mean) / std)

        return pd.DataFrame(normalized)

class ApplyFuncAdapter(VectorAdapter):
    def __init__(self, func=None):
        self._func = func or (lambda v : v)

    def adapt(self, series):
        return series.apply(self._func)

#Test methods

def testFile():
    file = r'D:\Guido\Master Finanzas\2018\Primer Trimestre\Metodos No Param\Base_Clientes Alemanes.xlsx'
    provider = ExcelDataProvider(file, 'Raw data')
    return provider.getData()

