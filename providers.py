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
            self.preComputeImp()
            self._loaded = True

    @abstractmethod
    def preComputeImp(self):
        pass

    @computeBefore
    def getData(self):
        return self._data


class ExcelDataProvider(Provider):
    def __init__(self, fileName, sheetName):
        super().__init__()
        self.fileName = fileName
        self.sheetName = sheetName

    def preComputeImp(self):
        print("Loading {}".format(self.fileName))
        self._data = pd.read_excel(self.fileName, self.sheetName)
        print("File loaded")


class AutoInferAdapterProvider(Provider):
    def __init__(self, provider, columns=None):
        super().__init__()
        self.provider = provider
        self.columns = columns

    def preComputeImp(self):
        self._data = self._adaptData()

    def _adaptData(self):
        data = self.provider.getData()[self.columns] if self.columns else self.provider.getData()
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
                return NumericNormalizedAdapter()
            except:
                return QualitativeToBinary()


class RawDataProvider(Provider):
    def __init__(self, provider):
        super().__init__()
        self.provider = provider

    def preComputeImp(self):
        self._data = self.provider.getData().values

# Adapters

class DataAdapter:
    def __init__(self, provider, adapterMapping):
        self.provider = provider
        self.adapterMapping = adapterMapping

    def adapt(self):
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

#Test methods

def testFile():
    file = r'D:\Guido\Master Finanzas\2018\Primer Trimestre\Metodos No Param\Base_Clientes Alemanes.xlsx'
    provider = ExcelDataProvider(file, 'Raw data')
    return provider.getData()

def testFilteredFile():
    file = r'D:\Guido\Master Finanzas\2018\Primer Trimestre\Metodos No Param\Base_Clientes Alemanes.xlsx'
    provider = ExcelDataProvider(file, 'Raw data')
    columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,17, 18, 19, 20]
    return FilteredProvider(provider,columns).getData()


def testAdapted():
    file = r'D:\Guido\Master Finanzas\2018\Primer Trimestre\Metodos No Param\Base_Clientes Alemanes.xlsx'
    provider = ExcelDataProvider(file, 'Raw data')
    mapping = {1: QualitativeToBinary(), 2: NumericNormalizedAdapter(), 3: QualitativeToBinary(),
               4: QualitativeToBinary(), 5: NumericNormalizedAdapter(),
               6: QualitativeToBinary()}
    adapter = DataAdapter(provider, mapping)
    return adapter

def testInferedAdapted():
    file = r'D:\Guido\Master Finanzas\2018\Primer Trimestre\Metodos No Param\Base_Clientes Alemanes.xlsx'
    provider = ExcelDataProvider(file, 'Raw data')
    columns = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    filteredProvider = FilteredProvider(provider, columns)
    return AutoInferAdapterProvider(filteredProvider)
