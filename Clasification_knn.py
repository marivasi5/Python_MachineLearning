import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats
from scipy.spatial import distance
import operator
from collections import Counter
import random
import os
os.system('wget http://mlearn.ics.uci.edu/databases/yeast/yeast.data')

def shuffle_listes_mazi(a,b):
    '''Anakatema antikeimenwn duo listwn, alla diatirisi antistoixou order'''
    combined = list(zip(a, b))
    random.Random(710).shuffle(combined)
    a[:], b[:] = zip(*combined)
    return a,b

def knn(x, label_x, y, k, dist):
    '''x: Matrix of training observations, y: input observation'''
    #ypoligismos twn apostasewn tou y me kathe simeio xi
    apostaseis=[]              #apothikeuei:(label, apostasi)      
    for xi, label_xi in zip(x.T, label_x):

        #epilogi distance metric
        if dist == 'euclidean':
            apostasi = distance.euclidean(xi, y)   
        elif dist == 'mahalanobis':
            cov= x.dot(x.T)                             
            theleiinverseomws=np.linalg.inv(cov)
            apostasi= scipy.spatial.distance.mahalanobis(y, xi, theleiinverseomws)
        elif dist == 'manhatan':
            apostasi = scipy.spatial.distance.cityblock(xi, y)   
        
        zeugarakia=[label_xi, apostasi] 
        apostaseis.append(zeugarakia)
    apostaseis.sort(key=operator.itemgetter(1))
    
    keepers=apostaseis[:(k)]         #krataw tis k pio kontines classes
    keepers_labels=[a[0] for a in keepers]
    counter= Counter(keepers_labels).most_common(1)
    #IN CASE OF TIES: Uses a random neighbor
    prediction=counter[0][0]
    return prediction

def accuracy(predictions, known_labels):
    bingo=0
    for a,b in zip(predictions, known_labels):        
        if a==b:
            bingo+=1
    accuracy = bingo / len(predictions)
    return accuracy

def leave_one_knn(X, labels, k, dist):
    D, N = X.shape
    predictions=[]
    known_labels=[]
    for i in range(N):
        y=X[:,i]
        x=np.delete(X, [i],axis=1)
        label_y=labels[i]
        known_labels.append(label_y)
        label_x= labels[:i] + labels[i+1 :]
    
        prediction=knn(x, label_x, y, k,dist) 
        predictions.append(prediction)
    return (predictions, known_labels)

def k_fold_knn(X, labels, k, kfold,  dist):
    '''Dinei kateu8eian accuracy! '''
    D, N = X.shape
    kommati= int(N/kfold)
    
    pairs=[]
    per_fold_accuracy=[]
    for i in range(0, N-(N-kommati*10), kommati):
        
        #sximatismos tou fold
        y=X[:,i:i+kommati]       
        x=np.hstack((X[:,0:i], X[:, i+kommati:]))
        
        known_labels=labels[i:i+kommati]
        label_x= labels[:i] + labels[i+kommati:]
        
        fold_predictions=[]
        for i in range(kommati):
            yi=y[:,i]
            prediction=knn(x, label_x, yi, k, dist) 
            fold_predictions.append(prediction)
        zeugaraki=(fold_predictions, known_labels)
        pairs.append(zeugaraki)
    
        fold_accuracy=accuracy(fold_predictions, known_labels)
        per_fold_accuracy.append(fold_accuracy)
        
    accuracara=sum(per_fold_accuracy) / float(len(per_fold_accuracy))
    return accuracara

def plot_leave(arxeio, dist):
    times_k, times_ac, times_xronou= zip(*arxeio)  
    
    megisti = max(times_ac)
    index= [i for i, j in enumerate(times_ac) if j == megisti][0]
    kalutero_k= times_k[index]
    print('Leave one out validation: Optimal k={} for {} distance (Diarkeia={} sec)'.format(kalutero_k, dist, round(sum(times_xronou),2)))

    plt.plot(times_k, times_ac,markersize=4, color='purple', alpha=0.5)
    plt.plot(kalutero_k, megisti,'*', markersize=5, color='blue', alpha=0.5)
    plt.xlabel('K-neighbors')
    plt.ylabel('Accuracy')
    plt.suptitle('Knn clasification accuracy for diferent k values', fontweight='bold')
    plt.title('Distance = {} - Validation = Leave-one-out'.format(dist))
    plt.savefig('knn_leave_{}'.format(dist))
    plt.show()
 
def plot_kfold(arxeio, dist):
    times_k, times_ac, times_xronou= zip(*arxeio)  
    
    megisti = max(times_ac)
    index= [i for i, j in enumerate(times_ac) if j == megisti][0]
    kalutero_k= times_k[index]
    print('10-fold validation: Optimal k={} for {} distance (Diarkeia={} sec)'.format(kalutero_k, dist, round(sum(times_xronou),2)))

    plt.plot(times_k, times_ac,markersize=4, color='blue', alpha=0.5)
    plt.plot(kalutero_k, megisti,'*', markersize=5, color='magenta', alpha=0.5)
    plt.xlabel('K-neighbors')
    plt.ylabel('Accuracy')
    plt.suptitle('Knn clasification accuracy for diferent k values', fontweight='bold')
    plt.title('Distance = {} - Validation = 10-Fold'.format(dist))
    plt.savefig('knn_kfold_{}'.format(dist))
    plt.show()
#%%                            DATA
with open('yeast.data') as file:
    data= []
    names=[]
    labels=[]                                   
    for line in file:
        line=line.rstrip('\n')
        splittedline= line.split()
        #name= splittedline[0]                        ;names.append(name)
        label= splittedline[9]                        ;labels.append(label)
        values= list(map(float, splittedline[1:9]))   ;data.append(values)
        
#anakatema dataset        
labels, data= shuffle_listes_mazi(labels, data)        

#ta atributes 5 kai 10 exoun tin idia timi se ola ta observations, opote den prosthetoun kapoia pliroforia kai afairountai
clean_data=[]
for lista in data:
    lista=lista[:4]+lista[6:]
    clean_data.append(lista)

pinakas=np.asmatrix(clean_data).T

#apomonwsi test set 10%
d, n =pinakas.shape
out=int(0.1*n)
test= pinakas[:, :out]
labels_test=labels[:out]

#apomonwsi training set X 
X=pinakas[:, out:]
labels=labels[out:]

#%%                             LEAVE ONE OUT-RUN
print('Starting validation runs...')
for dist in ['manhatan','euclidean', 'mahalanobis' ]:
    arxeio=[]
    for k in range(10, 81, 5):
        start=time.time()
        a= leave_one_knn(X, labels, k, dist)
        leave_one_out_predictions, known_labels =a
        akriveia=accuracy(leave_one_out_predictions, known_labels)
        xronos= time.time()-start
        keep= (k, akriveia, xronos)
        arxeio.append(keep)
        print('Leave one out Validation', dist, k, 'Done!')
    
    plot_leave(arxeio, dist)

#%%                               K-FOLD RUN 
kfold=10 
for dist in ['manhatan', 'mahalanobis', 'euclidean']:
    arxeio_k=[]
    for k in range(10, 100, 5):
        start=time.time()
        akriveia= k_fold_knn(X, labels, k, kfold, dist)
        xronos= time.time()-start
        keep= (k, akriveia, xronos)
        arxeio_k.append(keep)
        print('Kfold validation', dist, k, 'Done!')
    plot_kfold(arxeio_k, dist)

#%%                    TEST
Dtest, Ntest = test.shape
test_predictions=[]
for i in range(Ntest):
    y=test[:,i]
    label_y=labels_test[i]
    test_prediction=knn(X, labels, y, 20, 'manhatan')
    test_predictions.append(test_prediction)
test_accuracy=accuracy(test_predictions, labels_test)
print('Knn: Test accuracy= {}'.format(round(test_accuracy, 3)))



    
                    