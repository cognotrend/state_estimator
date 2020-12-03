"""Covariance"""
import numpy as np
class Covariance():
    '''
    The Covariance class implements a covariance matrix as a diagonal matrix of positive values
    '''
    def __init__(self, 
                 size=3, 
                 sigma1=0.001, 
                 sigma2=0.2, 
                 sigma3=0.4, 
                 rflag=0, 
                 msg = ''):
        '''
        Constructor for Covariance class.  Declare object as follows:
        myObject = covariance.Covariance()
        If you imported covariance as an alias, say, as cov, then an object declaration might be
        myObject = cov.Covariance()
        Parameters
        ----------
        size: size of the matrix, default=3
        sigma1:  parameters for three diagonal elements of default covariance
        sigma2
        sigma3
        The data member for this class is called Cov
        '''
        if size==1:
            self.Cov=np.array([[sigma1**2]],float)
        elif size==3:
            self.Cov = np.diag([sigma1**2,sigma2**2,sigma3**2])
        else:
            d=sigma1*np.ones(size)
            self.Cov = np.diag(d)
        self.size = size
        print(self.__doc__)
        print('\t\tImplementing: ',msg)

    def test(self):
        print(self.Cov)
        print(np.linalg.eig(self.Cov))

    def randomize(self):
        n=self.size
        X=np.random.rand(n,n)
        print(X)
        Ignore,U = np.linalg.eig((X + np.transpose(X))/2)
        print(U)
        self.Cov = U * np.diag(abs(np.random.randn(3, 1))) * np.transpose(U)


#myobj = Covariance()
#%print(myobj.__doc__)