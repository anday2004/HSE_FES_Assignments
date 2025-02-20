from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score



class RFFPipeline(BaseEstimator, TransformerMixin):
    """
    Пайплайн, делающий последовательно три шага:
        1. Применение PCA
        2. Применение RFF
        3. Применение классификатора
    """
    def __init__(self, n_features=1000, new_dim=50, use_PCA=True, classifier='logreg', kernel = 'linear'):
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        self.classifier = classifier
        self.kernel = kernel
        
    def fit(self, X, y):
        X_fit = X.copy()
        if self.use_PCA == True:
            self.dec = PCA(self.new_dim)
            X_fit = self.dec.fit_transform(X_fit)
        
        indices = np.random.randint(0, X_fit.shape[0], size=(1000000, 2))
        X_indices = X_fit[indices]
        sigma = np.sqrt(np.median(np.sum(np.square(X_indices[:, 0] - X_indices[:, 1]), axis = 1)))

        self.w = np.random.normal(0, (1 / sigma), [X_fit.shape[1], self.n_features])
        self.b = np.random.uniform(-np.pi, np.pi, self.n_features)
        phi = np.cos(X_fit @ self.w + self.b)

        if self.classifier == 'logreg':
            self.model = LogisticRegression()
        elif self.classifier == 'svm':
            self.model = SVC(kernel = self.kernel)
        elif self.classifier == 'linreg':
            self.model = LinearRegression()
        self.model.fit(phi, y)

        return self


    def predict_proba(self, X):
        X_pred = X.copy()
        if self.use_PCA == True:
            X_pred = self.dec.transform(X_pred)

        phi = np.cos(X_pred @ self.w + self.b)

        return self.model.predict_proba(phi)
        
        
    def predict(self, X):
        X_pred = X.copy()

        if self.use_PCA == True:
            X_pred = self.dec.transform(X_pred)

        phi = np.cos(X_pred @ self.w + self.b)
        return self.model.predict(phi)
    

class ORFPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, n_features=1000, new_dim=50, use_PCA=True, classifier='logreg', kernel = 'rbf'):
        self.n_features = n_features
        self.use_PCA = use_PCA
        self.new_dim = new_dim
        self.classifier = classifier
        self.kernel = kernel
        
    def fit(self, X, y):
        X_fit = X.copy()
        if self.use_PCA == True:
            self.dec = PCA(self.new_dim)
            X_fit = self.dec.fit_transform(X_fit)
        
        indices = np.random.randint(0, X_fit.shape[0], size=(1000000, 2))
        X_indices = X_fit[indices]
        sigma = np.sqrt(np.median(np.sum(np.square(X_indices[:, 0] - X_indices[:, 1]), axis = 1)))

        d = self.n_features // X_fit.shape[1]

        self.w = []
        self.b = np.random.uniform(-np.pi, np.pi, self.n_features)
        for _ in range(d):
            g = np.random.normal(0, 1, size=(X_fit.shape[1], X_fit.shape[1]))
            q, r = np.linalg.qr(g)
            s = np.diag(np.sqrt(np.random.chisquare(X_fit.shape[1], X_fit.shape[1])))
            w = (1 / sigma) * s @ q
            self.w.append(w)
        
        self.w = np.vstack(self.w)
        X_fit = X_fit[:, :self.n_features]
        self.phi = np.cos(X_fit @ self.w.T + self.b)
        
        if self.classifier == 'logreg':
            self.model = LogisticRegression()
        elif self.classifier == 'svm':
            self.model = SVC(kernel = self.kernel)
            
        self.model.fit(self.phi, y)

        return self


    def predict_proba(self, X):
        X_pred = X.copy()
        if self.use_PCA == True:
            X_pred = self.dec.transform(X_pred)

        X_pred = X_pred[:, :self.n_features]
        phi = np.cos(X_pred @ self.w.T + self.b)

        return self.model.predict_proba(phi)
        
        
    def predict(self, X):
        X_pred = X.copy()

        if self.use_PCA == True:
            X_pred = self.dec.transform(X_pred)

        X_pred = X_pred[:, :self.n_features]    
        phi = np.cos(X_pred @ self.w.T + self.b)
        
        return self.model.predict(phi)