import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math
from scipy.spatial import distance

#DATA
mean1=np.array([1,1])
mean2=np.array([-1,-1])
I = np.eye(2)
np.random.seed(1)
x1 = np.random.multivariate_normal(mean1, 0.5* I, 220).T
x2 = np.random.multivariate_normal(mean2, 0.75* I, 280).T                                  
x=np.append(x1, x2, axis=1)                                  

D, N=x.shape
#subtract the mean from each column vector
mean = np.mean(x , axis=1)
mean_matrix = np.tile(np.asmatrix(mean).T, N)  
x=x-np.asarray(mean_matrix)
                                  
plt.plot(x[0,:],  x[1,:], 'o' , color='purple', alpha=0.5)
plt.xlim([-4,4])
plt.ylim([-4,4])                   
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('artificial data')
plt.savefig("artificial data")
plt.show()        

plt.plot(x[0,0:220], x[1,0:220], 'o', color='blue', alpha=0.5, label='first cluster')
plt.plot(x[0,220:500],  x[1,220:500], '^' , color='red', alpha=0.5, label='second cluster')
plt.xlim([-4,4])
plt.ylim([-4,4])                   
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('artificial data-clusters insight!')
plt.savefig("artificial data-clusters insight!")
plt.show()        

#%%
K=2
maxiter=1000
mi= np.random.randn(D,K)                #pinakas       
pi= [1/K] * K                           #lista
covlista= [np.eye(D) + 2e-4] * K        #lista me pinakes

lastloglikelihood= -math.inf                  
EM=True
count=0
while EM:
    count+=1
    print(count)    
    if count==maxiter:
        break
    
#------E STEP: gemisma tou pinaka me ta responsibilities g(Znk) (diastaseis k X N: gia kathe simeio (stili) dinei pithanotita ana katanomi (grammi))
    
    gama=np.zeros((K,N))
    for n in range(N):
        xn= x[:,n]
        lista = []  #exei tous k arithmites (apo tin gamma) gia kathe xn
        
        for kk in zip(pi, covlista, mi.T):
            pk, covk, mk= kk
            
            katanomi = scipy.stats.multivariate_normal(mean= mk, cov=covk)
            #http://stackoverflow.com/questions/11615664/multivariate-normal-density-in-python
            arithmitis= pk * katanomi.pdf(xn)     
            lista.append(arithmitis)
        paronomastis=sum(lista)
        
        for k in range(len(lista)):
            gama[k,n]= lista[k]/paronomastis
            
#------M STEP: Re-estimate the parameters using the current responsibilities---
    
    NtwnK=(np.sum(gama, axis=1)).tolist() # ta Nk gia oles tis -k- katanomes
    pi_new= [i/N for i in NtwnK]
    
    #                  MEAN
    mi_new_lista=[]                     #tha gemisei ana stili-dld ana katanomi
    for grammes in zip(gama, NtwnK):    #kathe grammi antistoixi se mia katanomi
        grammi, Nk= grammes
        athroisma=0    
        for n in range(len(grammi)):
            respo=grammi[n]
            xn= x[:,n]
            ginomeno=respo * xn
            athroisma+= ginomeno
        
        mk_new= athroisma/Nk
        mi_new_lista.append(mk_new)
        
    mi_new=np.asarray(mi_new_lista).T        
    
    #              COVARIANCE
    covlista_new=[]
    for ziparismenosxamos in zip(gama, NtwnK, mi_new.T):    #antistoixei se mia katanomi
        grammi, Nk, mik_new= ziparismenosxamos
    
        athroismatara=0    
        for n in range(len(grammi)):
            respo=grammi[n]
            xn= x[:,n]
            xnorm= xn-mik_new
            ginomenonara=respo * np.asmatrix(xn).T.dot(np.asmatrix(xn))
            athroismatara+= ginomenonara
        
        covk_new= np.asarray(athroismatara)/Nk
        covlista_new.append(covk_new)
        
    #            LOGLIKELIHOOD
    loglikelihood=0
    for n in range(N):
        xn= x[:,n]
        superposition=0     #einai gia to kathe xn
        for new in zip(pi_new, covlista_new, mi_new.T):
            pk_new, covk_new, mk_new= new
            
            katanomi = scipy.stats.multivariate_normal(mean= mk_new, cov=covk_new)
            kommateli= pk_new * katanomi.pdf(xn)
            superposition+= kommateli
            
        loglikelihood+=superposition
    print('log', loglikelihood)
         
   #           CONVERGENCE

    if loglikelihood - lastloglikelihood > 1e-4:
        lastloglikelihood = loglikelihood
        mi=mi_new
        covlista=covlista_new
        pi = pi_new    
    else:       
        EM=False
        keep_mi=mi_new
        keep_covlista=covlista_new
        keep_pi = pi_new  
        
#Re-estimate the responsibilities
keep_gama=np.zeros((K,N))
for n in range(N):
    xn= x[:,n]
    keep_lista = []  #exei tous k arithmites (apo tin gamma) gia kathe xn
    
    for kk in zip(keep_pi, keep_covlista, keep_mi.T):
        keep_pk, keep_covk, keep_mk= kk
        
        katanomi = scipy.stats.multivariate_normal(mean= keep_mk, cov=keep_covk)
        arithmitis= keep_pk * katanomi.pdf(xn)     
        keep_lista.append(arithmitis)
    paronomastis=sum(keep_lista)
    
    for k in range(len(keep_lista)):
        keep_gama[k,n]= keep_lista[k]/paronomastis

#%%                ASSIGNMENT
assignment=[]
for gn in gama.T:
    label = np.where(gn == gn.max())[0][0]
    assignment.append(label)


#%%                  PLOT

labels=np.asarray(assignment)
plt.figure()
reds = labels == 0
blues = labels == 1
green = labels == 2
plt.plot(x.T[reds, 0], x.T[reds, 1], "ro")
plt.plot(x.T[blues, 0], x.T[blues, 1], "bo")
plt.plot(x.T[green, 0], x.T[green, 1], "go")
plt.plot(keep_mi[0,:], keep_mi[1,:],'X',markersize=10, color='black')

plt.savefig('MOG-4')
plt.show()

