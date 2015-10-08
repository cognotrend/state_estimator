import numpy as np
class Covariance():
    def __init__(self,size=3,sigma1=0.05,sigma2=0.5,sigma3=0.07):
        if size==1:
            self.Cov=np.array([[sigma1]],float)
        elif size==3:
            self.Cov = np.diag([sigma1,sigma2,sigma3])
        else:
            d=sigma1*np.ones(size)
            self.Cov = np.diag(d)
