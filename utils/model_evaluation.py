from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Transformador para seleccionar características específicas del DataFrame.
    
    Parámetros:
    - features: Lista de nombres de columnas a seleccionar. Si es None, se seleccionan todas las columnas.
    """

    def __init__(self, features_index=None):
        self.features_index = features_index
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.features_index is None:
            return X
        return X[:, self.features_index]
