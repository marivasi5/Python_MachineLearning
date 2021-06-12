import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.stats
import math
from scipy.spatial import distance
import operator
from collections import Counter
from scipy import stats
import random
import os
os.system('wget http://mlearn.ics.uci.edu/databases/yeast/yeast.data')

def shuffle_listes_mazi(a,b):
    '''Anakatema antikeimenwn duo listwn, alla diatirisi antistoixou order'''
    combined = list(zip(a, b))
    random.Random(710).shuffle(combined)
    a[:], b[:] = zip(*combined)
    return a,b

def probability(x, mean, std):
    	ekthetis = math.exp(-(math.pow(x-mean,2)/(2*math.pow(std,2))))
    	return (1 / (math.sqrt(2*math.pi) * std)) * ekthetis
 
def naive_bayes(x, y, label_x):    
    '''x: Matrix of training observations, y: input observation
    SOS: Einai listes (listwn) kai oxi pinakes'''    
    
    suxnotites_labels=[label_x.count(i)/len(label_x) for i in unique_labels]    
    #                   separate data by class    
    my_dic={}                      #dimiourgia dictionary me keys ola ta labels
    for i in unique_labels:
        my_dic[i]=[]
        
    for vector, label in zip(x, labels):     #xwrisma twn vector se kathe label
        my_dic[label].append(vector)
    for key, values in my_dic.items():
        my_dic[key]=np.asmatrix(values).T
    
    #                Create MEAN + STD matrix 10 X 8
    means=[]                         #Kathe antikeimeno antistixi se mia apo tis unique klaseis(exoun idia seira)- periexei 8 times gia ta attributes
    stds=[]
    for i in my_dic.values():
        mo= np.mean(i, axis=1)
        means.append(mo)
        std=np.std(i, axis=1)
        stds.append(std)
        
    #                      PROBABILITY
    products=[]   #periexei ena ginomeno pithanotitas ANA klasi
    for mean_vector, std_vector in zip(means, stds):
        pithanotites=[]
        for mean, std, yi in zip(mean_vector, std_vector, y):
            try:
                pithanotita=probability(yi, mean, std)
            except:
                pithanotita=0

            pithanotites.append(pithanotita)
        product=np.prod(pithanotites)
        products.append(product)    

    #                   Assignment to class            
    teliko=[i*j for i,j in zip(products,suxnotites_labels)]     
    megisti = max(teliko)                        ##SOS FTIAKSE TIE--- RANDOM TWRA
    index= [i for i, j in enumerate(teliko) if j == megisti][0]
    prediction=unique_labels[index]
    return prediction 

def accuracy(predictions, known_labels):
    bingo=0
    for a,b in zip(predictions, known_labels):        
        if a==b:
            bingo+=1
    accuracy = bingo / len(predictions)
    return accuracy

def k_fold_bayes(clean_data, labels, kfold):
    '''Dinei kateu8eian accuracy! '''
    N = len(clean_data)
    kommati= int(N/kfold)                 
    
    pairs=[]
    per_fold_accuracy=[]
    
    for i in range(0, N-(N-kommati*10), kommati):
        #sximatismos tou fold
        y=clean_data[i:i+kommati]       
        x=clean_data[:i] + clean_data[i+kommati:]
        known_labels=labels[i:i+kommati]
        label_x= labels[:i] + labels[i+kommati:]
        
        fold_predictions=[]
        for i in range(kommati):
            yi=y[i]
            prediction=naive_bayes(x, yi,label_x) 
            fold_predictions.append(prediction)
        zeugaraki=(fold_predictions, known_labels)
        pairs.append(zeugaraki)
    
        fold_accuracy=accuracy(fold_predictions, known_labels)
        per_fold_accuracy.append(fold_accuracy)
        
    accuracara=sum(per_fold_accuracy) / float(len(per_fold_accuracy))
    return accuracara

def leave_one_bayes(clean_data, labels):    
    '''Dinei kateu8eian accuracy! '''
    N= len(clean_data)
                                
    predictions=[]
    known_labels=[]
    for i in range(N):
        x=clean_data[i+1:]
        y=clean_data[i]
        
        label_y=labels[i]
        known_labels.append(label_y)
        label_x= labels[:i] + labels[i+1 :]
        #suxnotites_labels=[label_x.count(i)/len(label_x) for i in unique_labels]    
        #einai mesa stin bayes
        prediction=naive_bayes(x, y, label_x)
        predictions.append(prediction)
    
    accuracara=accuracy(predictions, known_labels)
    return accuracara

#%%                            DATA
with open('yeast.data') as file:
    data= []
    names=[]
    labels=[]                                   
    for line in file:
        line=line.rstrip('\n')
        splittedline= line.split()
        name= splittedline[0]                         ;names.append(name)
        label= splittedline[9]                        ;labels.append(label)
        values= list(map(float, splittedline[1:9]))   ;data.append(values)
unique_labels= list(set(labels))

#anakatema dataset        
labels, data= shuffle_listes_mazi(labels, data)        

#ta atributes 5 kai 10 exoun tin idia timi se ola ta observations, opote den prosthetoun kapoia pliroforia kai afairountai
clean_data=[]
for lista in data:
    lista=lista[:4]+lista[6:]
    clean_data.append(lista)
    
#apomonwsi test set 10%
n =len(clean_data)
out=int(0.1*n)                          
test= clean_data[:out]
labels_test=labels[:out]

clean_data=clean_data[out:]                    
labels=labels[out:]

#%%                          VALIDATION
print('Starting validations...')
accuracy_leave= leave_one_bayes(clean_data, labels)
kfold=10
accuracy_kfold= k_fold_bayes(clean_data, labels, kfold)

print('Validation: leave-one-out accuracy: {}, kfold accuracy: {}'.format(round(accuracy_leave,2), round(accuracy_kfold,2)))

#%%                             TEST
test_predictions=[]
for i in range(len(test)):
    yi=test[i]
    test_prediction=naive_bayes(clean_data, yi,labels) 
    test_predictions.append(test_prediction)
test_accuracy=accuracy(test_predictions, labels_test)
print('Naive Bayes: Test accuracy= {}'.format(round(test_accuracy, 3)))



