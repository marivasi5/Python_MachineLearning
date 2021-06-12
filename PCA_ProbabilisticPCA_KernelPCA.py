import numpy as np
import random
import scipy
from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import mean_squared_error
from scipy import exp
from scipy.linalg import eigh
import os
#%%

def PCA_eigendecomposition(x,k):
    '''Performs Principal Component Analysis 
    input: x: a DxN MATRIX where the observations are stored as columns (N) and the attributes defined as rows (D)
    k: number of principal components
    output: a kXN matrix of the projected data'''
    
    D, N =x.shape
    
    #subtract the mean from each column vector
    mean = np.mean(x , axis=1)
    mean = np.tile(mean, N)  
    xnorm=x-mean
    
    #Compute the covariance matrix and its eigenvectors-eigenvalues
    cov=np.cov(xnorm)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    #sort the eigenvectors by decreasing eigenvalues
    pairs=[(np.abs(eigenvalues[i]), eigenvectors[:,i]) for i in range(len(eigenvalues))]
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    #Define a matrix U whose columns are the k eigenvectors in descending order
    top=[pairs[i][1].reshape(D,1) for i in range(0,k)]
    U=np.hstack(top)   #einai D epi k
    #==============================================================================
    # top1=[pairs[i][1] for i in range(0,k)]
    # U1=np.asarray(top1).T  #einai D epi k
    #==============================================================================

    #Project the data onto the new subspace
    projected=U.T.dot(x)
    return projected    

def PCA_SVD(x,k):
    '''Performs Principal Component Analysis 
    input: x: a DxN MATRIX where the observations are stored as columns (N) and the attributes defined as rows (D)
    k: number of principal components
    output: a kXN matrix of the projected data'''
    
    D, N =x.shape
        
    U,s,V = scipy.linalg.svd(x, full_matrices=False)   #o U einai D ei rank 
    #eigenvalues of the covariance matrix S=square singular values (s). Eigenvectors of S are the columns of the matrix U
    
    #sort the eigenvectors by decreasing eigenvalues
    pairs=[(np.abs(s[i]), U[:,i]) for i in range(len(s))]
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    #Define a matrix U1 whose columns are the k eigenvectors in descending order
    top=[pairs[i][1].reshape(D,1) for i in range(0,k)]
    U1=np.hstack(top)   #einai D epi k
    
    #Project the data onto the new subspace
    projected1=U1.T.dot(x)
    return projected1    

def PPCA(x, k, error1, error2):
    '''Performs Probabilistic Principal Component Analysis 
    input: x: a DxN MATRIX where the observations are stored as columns (N) and the attributes defined as rows (D)
    k: number of principal components
    error1: Root Mean Squared Error threshold for the W matrix
    error2: threshold for s2
    output: a kXN matrix of the projected data'''
    
    mean = np.mean(x , axis=1)    
    D, N=x.shape
    I = np.matrix(np.eye(k))         
    
    s_old = random.uniform(0,1)
    W_old = np.matrix(np.random.randn(D,k))
    
    if s_old > 0:
        def Estep(x, M, W_old, s_old):
            xnorm=x-mean
            Ezn = np.linalg.inv(M).dot(W_old.T).dot(xnorm)
            EznznT = s_old*np.linalg.inv(M) + Ezn.dot(Ezn.T)
            return xnorm , Ezn, EznznT    
        
        counter=0
        EM=True
        while EM:
            counter+=1
            print(counter)             

            A = np.asmatrix(np.zeros((D,k)))
            B = np.asmatrix(np.zeros((k,k)))
            sum=0
            M = W_old.T.dot(W_old) + s_old*I
            
            #Estimate maximum likelihood estimation for W
            for xn in x.T:  #gia na parw stili
                xn=xn.T     #i numpy kanei tin stili pou pira grammi, ara prepei na to toumparw
                
                xnorm, Ezn, EznznT = Estep(xn, M, W_old, s_old)        
                A+= xnorm.dot(Ezn.T)
                B+= EznznT
            W_new=A.dot(B.I)
                
            #Estimate maximum likelihood estimation for s2
            for xn in x.T:  #gia na parw stili
                xn=xn.T     #i numpy kanei tin stili pou pira grammi, ara prepei na to toumparw
                
                xnorm, Ezn, EznznT = Estep(xn, M, W_old, s_old)        
                treno= ((np.linalg.norm(xnorm))**2 - 2*(Ezn.T.dot(W_new.T).dot(xnorm)) + np.trace(EznznT.dot(W_new.T).dot(W_new)))[0,0]
                sum+=treno
            s_new= sum/(N*D)
            
            #Check for convergence    
            wmetric=(mean_squared_error(W_old, W_new))**0.5
            smetric=abs(s_old - s_new)
                       
            if wmetric < error1 and smetric <  error2:          
                W=W_new
                s=s_old
                EM=False
            else:
                W_old=W_new
                s_old=s_new
                
        #Perform singular value decomposition of W matrix
        U,S,V = np.linalg.svd(W, full_matrices=False)
        Projected = U.T.dot(x)
        
        return Projected

def Kernel_PCA_Gaussian(x, vita, k):
    '''Performs Kernel Principal Component Analysis  
    input: x: a DxN MATRIX where the observations are stored as columns (N) and the attributes defined as rows (D)
    k: number of principal components
    output: a kXN matrix of the transformed data'''
    
    D, N =x.shape
    
    #kernel matrix    
    K = np.asmatrix(np.zeros((N,N)))
    for i in range(N):
        xi= x[:,i]
        for j in range(N):
            xj = x[:,j]
            dist = np.linalg.norm(xi-xj)
            K[i,j] = exp(-vita*(dist**2))

    #Construct the normalized kernel matrix
    one_n = np.ones((N,N)) / N
    Knorm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # Obtaining eigenvalues in descending order with corresponding eigenvectors
    eigenvalues, eigenvectors = eigh(Knorm)

    #Define a matrix U whose columns are the k eigenvectors in descending order
    U= np.column_stack((eigenvectors[:,-i] for i in range(1,k+1)))
    
    return U

def Kernel_PCA_Polynomial(x, p, k):
    '''Performs Kernel Principal Component Analysis using Polynomial non-linear transformation 
    input: x: a DxN MATRIX where the observations are stored as columns (N) and the attributes defined as rows (D)
    k: number of principal components
    p: probability in range [0,1]
    output: a kXN matrix of the transformed data'''

    D, N =x.shape
    
    #kernel matrix    
    K = np.asmatrix(np.zeros((N,N)))
    for i in range(N):
        xi= x[:,i]
        for j in range(N):
            xj = x[:,j]
            eswt_ginomeno=xi.T.dot(xj)
            a=eswt_ginomeno[0,0]
            K[i,j] = (1+a)**p
                        
    #Construct the normalized kernel matrix
    one_n = np.ones((N,N)) / N
    Knorm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # Obtaining eigenvalues in descending order with corresponding eigenvectors
    eigenvalues, eigenvectors = eigh(Knorm)

    #Define a matrix U whose columns are the k eigenvectors in descending order
    U= np.column_stack((eigenvectors[:,-i] for i in range(1,k+1)))
    
    return U

def Kernel_PCA_Hyper(x, delta, k):
    '''Performs Kernel Principal Component Analysis  
    input: x: a DxN MATRIX where the observations are stored as columns (N) and the attributes defined as rows (D)
    k: number of principal components
    output: a kXN matrix of the transformed data'''
    import math
    from scipy.linalg import eigh
    D, N =x.shape
    
    #kernel matrix    
    K = np.asmatrix(np.zeros((N,N)))
    for i in range(N):
        xi= x[:,i]
        for j in range(N):
            xj = x[:,j]
            eswt_ginomeno=xi.T.dot(xj)
            a=eswt_ginomeno[0,0]
            K[i,j] = math.tan(a + delta)

    #Construct the normalized kernel matrix
    one_n = np.ones((N,N)) / N
    Knorm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
    
    # Obtaining eigenvalues in descending order with corresponding eigenvectors
    eigenvalues, eigenvectors = eigh(Knorm)

    #Define a matrix U whose columns are the k eigenvectors in descending order
    U= np.column_stack((eigenvectors[:,-i] for i in range(1,k+1)))
    
    return U
        
#%%                   SYNTHETIC DATA
X, y = make_circles(n_samples=1000, factor=.3, noise=.05)
x = np.asmatrix(X.T)
D, N =x.shape

plt.figure()
reds = y == 0
blues = y == 1
plt.plot(X[reds, 0], X[reds, 1], "ro")
plt.plot(X[blues, 0], X[blues, 1], "bo")
plt.savefig('synthetic_data.jpg')
plt.show()

#%%                       PCA
new_matrix=PCA_eigendecomposition(x, 2)

plt.plot(new_matrix.T[reds, 0], new_matrix.T[reds, 1], "ro")
plt.plot(new_matrix.T[blues, 0], new_matrix.T[blues, 1], "bo")
plt.title("PCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
plt.savefig('synt_PCA.jpg')
plt.show()

#%%                       PPCA
error1=10**(-6)
error2=10**(-6)
ProjectedData=PPCA(x, 2, error1, error2)

plt.plot(ProjectedData.T[reds, 0], ProjectedData.T[reds, 1], "ro")
plt.plot(ProjectedData.T[blues, 0], ProjectedData.T[blues, 1], "bo")
plt.title("PPCA")
plt.xlabel("1st principal component")
plt.ylabel("2nd principal component")
plt.savefig('synt_PPCA.jpg')
plt.show()

#%%                      KERNEL-GAUSSIAN
vita=3     
projected=Kernel_PCA_Gaussian(x, vita , 2)

plt.figure(figsize=(8,6))
plt.scatter(projected[y==0, 0], projected[y==0, 1], color='red', alpha=0.5)
plt.scatter(projected[y==1, 0], projected[y==1, 1], color='blue', alpha=0.5)
plt.title('Kernel PCA Gaussian')
plt.savefig("synthetic_kernel_gaussian_{}.jpg".format(vita))
plt.show()

#%%                    KERNEL-POLUNOMIAL 
p=5
projected=Kernel_PCA_Polynomial(x, p, 2)

plt.figure(figsize=(8,6))
plt.scatter(projected[y==0, 0], projected[y==0, 1], color='red', alpha=0.5)
plt.scatter(projected[y==1, 0], projected[y==1, 1], color='blue', alpha=0.5)
plt.title('Kernel PCA Polynomial')
plt.savefig("synthetic_kernel_polynomial{}.jpg".format(p))
plt.show()

#%%                     KERNEL-HYPERBOLIC TANGEN
delta=0.3     
projected=Kernel_PCA_Hyper(x, delta , 2)

plt.figure(figsize=(8,6))
plt.scatter(projected[y==0, 0], projected[y==0, 1], color='red', alpha=0.5)
plt.scatter(projected[y==1, 0], projected[y==1, 1], color='blue', alpha=0.5)
plt.title('Kernel PCA Hyperbolic Tangen')
plt.savefig("synthetic_kernel_hyper{}.jpg".format(delta))
plt.show()

#%%                        CHANGE DATASET NOISE
X, y = make_circles(n_samples=1000, factor=.3, noise=3)
x = np.asmatrix(X.T)
D, N =x.shape

plt.figure()
reds = y == 0
blues = y == 1
plt.plot(X[reds, 0], X[reds, 1], "ro")
plt.plot(X[blues, 0], X[blues, 1], "bo")
plt.savefig('synthetic_dataNOISE3.jpg')
plt.show()
#%%
vita=7    
projected=Kernel_PCA_Gaussian(x, vita , 2)

plt.figure(figsize=(8,6))
plt.scatter(projected[y==0, 0], projected[y==0, 1], color='red', alpha=0.5)
plt.scatter(projected[y==1, 0], projected[y==1, 1], color='blue', alpha=0.5)
plt.title('Kernel PCA Gaussian')
plt.savefig("syntheticNOISE3_kernel_gaussian_{}.jpg".format(vita))
plt.show()

#%%                  RANDOM DATASET
mean=np.ones(10)
I=np.eye(10)
x= np.random.multivariate_normal(mean, I, 10)
######################PERNA TA PLOTS#################################################

#%%                   PRACTICAL-CREATE MATRIX
os.system('wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248.soft.gz')
os.system('gunzip GDS6248.soft.gz')
os.system('cat GDS6248.soft | grep -v [!#^] | grep -v GSM > real.txt')

#%%

with open('/home/rantaplan/master/pca/real.txt') as file:
    pinakas=[]    
    for line in file:
        line=line.rstrip('\n')
        splittedline= line.split('\t')
        pinakas.append(splittedline[2:])
        
x = np.asmatrix(pinakas)
x = x.astype(np.float)
        
Projected=PPCA(x, 2, 10**(-3), 10**(-3))
#Τρεχει και γρηγορα απλα δεν προλαβα να κανω το πλοτ :)











