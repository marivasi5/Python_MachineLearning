import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics import silhouette_samples, silhouette_score
import math
import scipy
import time
import matplotlib.cm as cm
import os
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
#plt.savefig("artificial data")
plt.show()        

plt.plot(x[0,0:220], x[1,0:220], 'o', color='blue', alpha=0.5, label='first cluster')
plt.plot(x[0,220:500],  x[1,220:500], '^' , color='red', alpha=0.5, label='second cluster')
plt.xlim([-4,4])
plt.ylim([-4,4])                   
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('artificial data-clusters insight!')
#plt.savefig("artificial data-clusters insight!")
plt.show()
        
#%%
def my_kmeans(x, k, dist, maxiter, reps):
    D, N=x.shape
    
    rep_min= math.inf
    for i in range(0, reps):
        
        centroids= np.random.rand(D,k)                                  
        count=0
        trexa=True
        while trexa:
            count+=1
            #print(count)                                 
            if count==maxiter:
                break
                
    ##                       ASSIGNMENT STOIXEIWN
            #apostaseis=[]       
            assignment=[]
            for xi in x.T:
                apostasi= math.inf
                
                c_pos=0           #tha pernei tin timi tou aithmou stilis apo ton pinaka twn centroids
                for ci in centroids.T:
                    #epilogi distance metric
                    if dist == 'euclidean':
                        apostasi_new = distance.euclidean(xi, ci)   
                    elif dist == 'mahalanobis':
                        cov= x.dot(x.T)                             
                        theleiinverseomws=np.linalg.inv(cov)
                        apostasi_new= scipy.spatial.distance.mahalanobis(ci, xi, theleiinverseomws)
                    elif dist == 'manhatan':
                        apostasi_new = scipy.spatial.distance.cityblock(xi, ci)   
                    
                    if apostasi_new < apostasi:
                        apostasi=apostasi_new  
                        c=c_pos
                    c_pos+=1
                assignment.append(c)
                #apostaseis.append(apostasi)    
            
    ##                   SXIMATISMOS CLUSTER        
    
            clusters_dict={}         #tha valei key=label /value=ta stoixeia tou clusters
            for number_of_cluster in range(0,k):
                clusters_dict[number_of_cluster]= []
                for index in range(0, len(assignment)):    
                    if assignment[index]==number_of_cluster:
                        clusters_dict[number_of_cluster].append(x[:,index])    
                
                clusters_dict[number_of_cluster]=np.array(clusters_dict[number_of_cluster]).T
            
    ##                        NEW CENTROIDS
            centroids_new_list=[]                 
            for pinaka in clusters_dict.values():
                centroids_new_list.append(np.mean(pinaka, axis=1))               
                             
            centroids_new = np.array(centroids_new_list).T
            ####centroids_new = [np.mean(x, axis=0) for x in clusters]
                    
    ##                         CONVERGENCE
            if (centroids==centroids_new).all():
                trexa=False
            else:
                centroids=centroids_new
    
    ####                    INTRACLUSTER DISTANCES
        cluster_means=[]
        for number_of_cluster, cluster in clusters_dict.items():
            ci=centroids[:,0]
            intra=[]
            for xi in cluster.T:
                if dist == 'euclidean':
                   apostasoula= distance.euclidean(xi, ci)   
                elif dist == 'mahalanobis':
                    cov= x.dot(x.T) 
                    theleiinverseomws=np.linalg.inv(cov)
                    apostasoula = scipy.spatial.distance.mahalanobis(ci, xi, theleiinverseomws)     
                elif dist == 'manhatan':
                    apostasoula= scipy.spatial.distance.cityblock(xi, ci)   
                            
                intra.append(apostasoula)         #lista me intracluster apostaseis
                intra_mean= sum(intra)/len(intra)
        cluster_means.append(intra_mean)       #lista me k stoixeia
        rep_sum=sum(cluster_means)
        
        # hold the replication with lowest total distance within clusters
        if rep_sum < rep_min:
            rep_min=rep_sum
            keep_centroids= centroids
            keep_clusters_dict= clusters_dict
            keep_assignment= assignment
            
    return(keep_centroids, keep_assignment, keep_clusters_dict)

#%%
def my_plot(clusterakia_list, centroids_new, dist, reps, xronos):
    K = len(clusterakia_list)
    plt.plot(clusterakia_list[0][0,:], clusterakia_list[0][1,:], 'o', markersize=7, color='blue', alpha=0.5, label='First Cluster')
    plt.plot(clusterakia_list[1][0,:], clusterakia_list[1][1,:], 'o', markersize=7, color='purple', alpha=0.5, label='Second Cluster')
    if K > 2:
        plt.plot(clusterakia_list[2][0,:], clusterakia_list[2][1,:], 'o', markersize=7, color='red', alpha=0.5, label='Third Cluster')

    if K > 3:
        plt.plot(clusterakia_list[3][0,:], clusterakia_list[3][1,:], 'o', markersize=7, color='magenta', alpha=0.5, label='Forth Cluster')
    
    if K > 4:
        plt.plot(clusterakia_list[4][0,:], clusterakia_list[4][1,:], 'o', markersize=7, color='orange', alpha=0.5, label='Fifth Cluster')
    
    plt.plot(centroids_new[0,:], centroids_new[1,:],'X',markersize=10, color='black')
    #plt.xlim([-4,4])
    #plt.ylim([-4,4])                    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.legend(loc='lower right')
    plt.title('Kmeans Clustering \n Distance={}, K={} \n Running time for {} repetitions = {}s'.format(dist, K, reps, xronos))
    plt.savefig('Kmeans Clustering (K={}, Distance ={} )'.format(K, dist))
    plt.show() 


#%%                       TREKSIMO

maxiter=500
reps=10

run=[]
for k in range(2,6):
    for dist in ['manhatan', 'mahalanobis', 'euclidean']:        
        start=time.time()
        centroids_new, assignment, clusters_dict = my_kmeans(x, k, dist, maxiter, reps)
        xronos=round(time.time()-start, 2)        
        info = (k, dist, xronos)
        run.append(info)
        #                        GET MATRICES FROM DICT
        clusterakia_list=[]
        for i in clusters_dict.values():
            clusterakia_list.append(i)
        
        my_plot(clusterakia_list, centroids_new, dist, reps, xronos)

print(run)

#%%
#==============================================================================
# #%%                  ALLOS TROPOS GIA PLOT
# assignment=np.asarray(assignment)
# plt.figure()
# 
# purple = assignment == 0
# red = assignment == 1
# blue = assignment == 2
# magenta = assignment == 3
# yellow = assignment == 4
# green = assignment == 5
# 
# plt.plot(x.T[purple, 0], x.T[purple, 1],  'o', markersize=7, color='purple', alpha=0.5, label='First Cluster')
# plt.plot(x.T[red, 0], x.T[red, 1],  'o', markersize=7, color='red', alpha=0.5, label='Second Cluster')
# plt.plot(x.T[blue, 0], x.T[blue, 1], 'o', markersize=7, color='blue', alpha=0.5, label='Third Cluster')
# plt.plot(x.T[magenta, 0], x.T[blue, 1], 'o', markersize=7, color='magenta', alpha=0.5, label='Forth Cluster')
# plt.plot(x.T[yellow, 0], x.T[yellow, 1], 'o', markersize=7, color='yellow', alpha=0.5, label='Fifth Cluster')
# plt.plot(x.T[green, 0], x.T[green, 1], 'o', markersize=7, color='green', alpha=0.5, label='Fifth Cluster')
# plt.plot(centroids_new[0,:], centroids_new[1,:],'X',markersize=10, color='black')
# plt.show()
# 
#==============================================================================

#%%                   SILOUETTE

maxiter=500
reps=10

arxeio=[]
for dist in ['manhatan', 'mahalanobis', 'euclidean']:
    for k in range(2,3):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        centroids_new, assignment, clusters_dict = my_kmeans(x, k, dist, maxiter, reps)
    
        silhouette_avg = silhouette_score(x.T, np.asarray(assignment), metric='euclidean')
        print('Distance = {}. The average silhouette_score for K={} is {} '.format(dist, k, round(silhouette_avg, 3)))
        plirofories=(dist, k, round(silhouette_avg, 3))
        arxeio.append(plirofories)
        
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(x.T, np.asarray(assignment), metric='euclidean')
    
        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[np.asarray(assignment) == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.spectral(float(i) / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("Silhouette plot")
        ax1.set_xlabel("silhouette coefficient values")
        ax1.set_ylabel("K")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        # 2nd Plot showing the clusters formed
        colors = cm.spectral(np.asarray(assignment).astype(float) / k)
        ax2.scatter(x.T[:, 0], x.T[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)
    
        ax2.scatter(centroids_new[0,:], centroids_new[1,:], marker='X', c="black", alpha=1, s=50)
    
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        plt.suptitle(("Silhouette analysis for Kmeans Clustering (K={})".format(k)), fontsize=14, fontweight='bold')
        plt.savefig('Distance = {}.Silhouette analysis (K={}).jpeg'.format(dist, k))
        plt.show()
    
    
#%%==============================================================================
#----------------------PRACTICAL-------------------------------------------------
#%%==============================================================================

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

#%%                   PRACTICAL-CREATE MATRIX

os.system('wget ftp://ftp.ncbi.nlm.nih.gov/geo/datasets/GDS6nnn/GDS6248/soft/GDS6248.soft.gz')
os.system('gunzip GDS6248.soft.gz')
os.system('cat GDS6248.soft | grep -v [!#^] | grep -v GSM > real.txt')

#%%                 MEIWSI DIASTASEWN ME PPCA
with open('/home/rantaplan/master/pca/real.txt') as file:
    pinakas=[]    
    for line in file:
        line=line.rstrip('\n')
        splittedline= line.split('\t')
        pinakas.append(splittedline[2:])
        
x = np.asmatrix(pinakas)
x = x.astype(np.float)
        
Projected=PPCA(x, 2, 10**(-3), 10**(-3))
x= np.asarray(Projected)
#%%

from matplotlib import pyplot as plt

labels = [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
    
triplets_ppca = []

for i in range(Projected.shape[1]):
    triplet = [Projected[0,i], Projected[1,i], labels[i]]
    triplets_ppca.append(triplet)
    
ppca1 = [x[0] for x in triplets_ppca if x[2]==0]
ppca2 = [x[1] for x in triplets_ppca if x[2]==0]
ppca3 = [x[2] for x in triplets_ppca if x[2]==0]
ppca4 = [x[0] for x in triplets_ppca if x[2]==1]
ppca5 = [x[1] for x in triplets_ppca if x[2]==1]
ppca6 = [x[2] for x in triplets_ppca if x[2]==1]
ppca7 = [x[0] for x in triplets_ppca if x[2]==2]
ppca8 = [x[1] for x in triplets_ppca if x[2]==2]
ppca9 = [x[2] for x in triplets_ppca if x[2]==2]


plt.figure()
plt.scatter(ppca1, ppca2, c='red',alpha=0.5, label='baseline')
plt.scatter(ppca4, ppca5, c='purple',alpha=0.5, label='normal')
plt.scatter(ppca7, ppca8, c='blue',alpha=0.5, label='high-fat')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title("C57BL/6J mice")
plt.legend()
plt.savefig("PPCA.jpeg")
plt.show

#%%

maxiter=500
reps=10

arxeio=[]
for dist in ['manhatan', 'mahalanobis', 'euclidean']:
    for k in range(2,6):
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        centroids_new, assignment, clusters_dict = my_kmeans(x, k, dist, maxiter, reps)
    
        silhouette_avg = silhouette_score(x.T, np.asarray(assignment), metric='euclidean')
        print('Distance = {}. The average silhouette_score for K={} is {} '.format(dist, k, round(silhouette_avg, 3)))
        plirofories=(dist, k, round(silhouette_avg, 3))
        arxeio.append(plirofories)
        
        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(x.T, np.asarray(assignment), metric='euclidean')
    
        y_lower = 10
        for i in range(k):
            # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[np.asarray(assignment) == i]
    
            ith_cluster_silhouette_values.sort()
    
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i
    
            color = cm.spectral(float(i) / k)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
    
            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    
            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
    
        ax1.set_title("Silhouette plot")
        ax1.set_xlabel("silhouette coefficient values")
        ax1.set_ylabel("K")
    
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    
        # 2nd Plot showing the clusters formed
        colors = cm.spectral(np.asarray(assignment).astype(float) / k)
        ax2.scatter(x.T[:, 0], x.T[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)
    
        ax2.scatter(centroids_new[0,:], centroids_new[1,:], marker='X', c="black", alpha=1, s=50)
    
        ax2.set_xlabel("x1")
        ax2.set_ylabel("x2")
        plt.suptitle(("Silhouette analysis for Kmeans Clustering (K={})".format(k)), fontsize=14, fontweight='bold')
        plt.savefig('Distance = {}.Silhouette analysis (K={}).jpeg'.format(dist, k))
        plt.show()



        