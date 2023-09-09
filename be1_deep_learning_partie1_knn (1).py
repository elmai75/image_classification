# -*- coding: utf-8 -*-
"""BE1_deep_learning_partie1_knn.ipynb
la définition de la fonction unpickle qui prend le chemin du fichier et qui retourne le dictionnaire qui contient 4 clès
"""

#la définition de la fonction unpickle
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

file='C:/Users/elmai/Downloads/data/data_batch_1'

data=unpickle(file)

#affichages des clé du dictionnaire
data.keys()

data[b'data']

import numpy as np
#la definition de la fonction read_cifar_batch pour lire un seul batch
def read_cifar_batch(file,k=0):
  batch=unpickle(file)
  if k==0:
    return batch[b'data']
  if k==1:
    return np.array(batch[b'labels'])

#la définition de la fonction read_cifar qui va lire toutes les data_batch et le test_batch
from os import walk

def read_cifar(path,k=0):
  filenames = next(walk(path), (None, None, []))[2]
  p1=path+'/'+filenames[0]
  p2=path+'/'+filenames[1]
  p3=path+'/'+filenames[2]
  p4=path+'/'+filenames[3]
  p5=path+'/'+filenames[4]
  p6=path+'/'+filenames[5]

  data1=read_cifar_batch(p1,k=0)
  data2=read_cifar_batch(p2,k=0)
  data3=read_cifar_batch(p3,k=0)
  data4=read_cifar_batch(p4,k=0)
  data5=read_cifar_batch(p5,k=0)
  data6=read_cifar_batch(p6,k=0)
  concate1=np.concatenate((data1,data2,data3))
  concate2=np.concatenate((data4,data5,data6))
  concate=np.concatenate((concate1,concate2))

  label1=read_cifar_batch(p1,k=1)
  label2=read_cifar_batch(p2,k=1)
  label3=read_cifar_batch(p3,k=1)
  label4=read_cifar_batch(p4,k=1)
  label5=read_cifar_batch(p5,k=1)
  label6=read_cifar_batch(p6,k=1)
  lb1=np.concatenate((label1,label2,label3))
  lb2=np.concatenate((label4,label5,label6))
  lb=np.concatenate((lb1,lb2))

  if k==0:
    return concate
  if k==1:
    return lb

#la defintion de la fonction split_dataset
from numpy.random import PCG64
import numpy as np
from sklearn.model_selection import train_test_split
def split_dataset(data,label,split,k=0):
  data_train, data_test, labels_train, labels_test = train_test_split(data,label, test_size=split, random_state=42)
  if k==0:
    return data_train
  if k==1:
    return data_test
  if k==2:
    return labels_train
  if k==3:
    return labels_test

"""On va tester la fonction split_dataset sur une seule batch avec la repartirion de la partie d'entrainement et de test en 80% et 20%"""

# on va appliquer la fonction split sur une seule batch
data_train_batch=split_dataset(read_cifar_batch(file,0),read_cifar_batch(file,1),0.2,0)
data_test_batch=split_dataset(read_cifar_batch(file,0),read_cifar_batch(file,1),0.2,1)
labels_train_batch=split_dataset(read_cifar_batch(file,0),read_cifar_batch(file,1),0.2,2)
labels_test_batch=split_dataset(read_cifar_batch(file,0),read_cifar_batch(file,1),0.2,3)

"""la verification des dimension de la data_test et data_train"""

x,y=data_test_batch.shape,data_train_batch.shape
x,y

"""Mainteneant on va appliquer la fonction split_dataset sur tout le data_set en appelant la methode read_ciraf avec une répartition de 10% 90%"""

path='C:/Users/elmai/Downloads/data'
data_train=split_dataset(read_cifar(path,0),read_cifar(path,1),0.1,0)
data_test=split_dataset(read_cifar(path,0),read_cifar(path,1),0.1,1)
labels_train=split_dataset(read_cifar(path,0),read_cifar(path,1),0.1,2)
labels_test=split_dataset(read_cifar(path,0),read_cifar(path,1),0.1,3)

"""la verification des dimension de la data_test et data_train"""

x,y=data_test.shape,data_train.shape
print(x,y)

"""Dans cette partie on va définir la fonction distance matricielle"""

# Premièrment on va définir la focntion dist_matrix_batch qui calcule juste la distance entre 2 vecteur
def dist_matrix_batch(a,b):
  return np.dot(a,a.transpose())+np.dot(b,b.transpose())-2*np.dot(a,b.transpose())

"""la fonction dist_matrix calcule la distance matricielle"""

from scipy.optimize.slsqp import concatenate
def dist_matrix(data_train,data_test):
  concate=dist_matrix_batch(data_train[:data_test.shape[0]],data_test)
  l=int(data_train.shape[0]/data_test.shape[0])
  for i in range(1,l):
    m=dist_matrix_batch(data_train[data_test.shape[0]*i:data_test.shape[0]*(i+1)],data_test)
    concate=np.concatenate((concate,m),axis=1)
  return concate

"""Le précedent algorithme de calcule de la distance matricielle necessite un temps d'exécution très lent sur toute le data_set qui peut durer plusieurs heures, c'est pour cela j'ai choisie de l'appliquer sur un seule batch"""

l=dist_matrix(data_train_batch,data_test_batch)

l.shape

"""On va définir la fonction most_frequent qui va retourner l'élement le plus fréquenté dans une liste , que l'on va utiliser oar suite pour prédir la classe associé à un élément dans un nuage de k point."""

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

"""Dans cette etape on va créer la fonction predict-knn qui va retourner les labels du test set"""

def knn_predict(dists,label_train,k):
  pred=[]
  for i in range(dists.shape[0]):

    m=sorted(range(len(dists[i,:])), key = lambda sub: dists[i,:][sub])[:k]
    pred.append(most_frequent(list(label_train[m])))
  return pred

"""On va calculer les valeurs prédites du data_test à l'aide de la fonction knn_predict pour un seul batch et avec une valeur de k=10"""

valeur_predites=knn_predict(l,labels_train_batch,10)

len(valeur_predites)

#affichage des valeurs prédites
valeur_predites

#calcule de l'accuracy pour cette exemple
r=list(labels_test_batch)
ac=0
for i in range(labels_test_batch.shape[0]):
  if(r[i]==valeur_predites[i]):
    ac+=1
  
print(ac/len(r))

"""On remarque que l'accuracy est faible ne dépasse pas 10%

Création de la fonction evaluate_knn qui prend comme paramètres data_train,label_train,data_test,label_test,k et qui retourne l'accuracy
"""

def evaluate_knn(data_train,label_train,data_test,label_test,k):
  dist=dist_matrix(data_train,data_test)
  knn_predict(dist,label_train,k)
  r=list(label_test)
  ac=0
  for i in range(label_test.shape[0]):
    if(r[i]==valeur_predites[i]):
      ac+=1
  return(ac/len(r))

"""On va appliquer la fonction evaluaate knn sur notre batch , puis on va créer une liste qui contient la variantion d'accuracy en fonction de k"""

all_eval=[]
for i in range(10,20,1):
  eval=evaluate_knn(data_train_batch,labels_train_batch,data_test,labels_test_batch,i)
  all_eval.append(eval)

import matplotlib.pyplot as plt
plt.plot(all_eval)

"""On va utiliser une deuxième méthode qui est basé sur la definition de modèle dans une classe"""

class KNN_model(object):
    #definir le constructeur de la classe
    def __init__(self):
        pass
    # définir l'entranement de la classe 
    def train(self, X, y):
        self.X_train = X
        self.y_train = y
    def predict(self, X, k=1, num_loops=0):
        if num_loops == 0:
            dists = self.compute_distances(X)
        else:
            raise ValueError('Valeur invalide %d pour num_loops' % num_loops)
        return self.predict_labels(dists, k=k)
    #calcule de la distance euclidienne
    def compute_distances(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train)) 
        dists = np.sqrt(np.sum(np.square(self.X_train), axis=1) + np.sum(np.square(X), axis=1)[:, np.newaxis] - 2 * np.dot(X, self.X_train.T))
        pass
        return dists
    #prédiction 
    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            sorted_dist = np.argsort(dists[i])
            closest_y = list(self.y_train[sorted_dist[0:k]])
            pass
            y_pred[i]= (np.argmax(np.bincount(closest_y)))
            pass
        return y_pred

"""6- Le calcul de l'accuracy de notre modèle KNN sur les données test avec un k=5"""

num_test= labels_test.shape[0]
classifier = KNN_model()
classifier.train(data_train, labels_train)
dists= classifier.compute_distances(data_test)
y_test_pred = classifier.predict_labels(dists, k=5)
num_correct = np.sum(y_test_pred == labels_test)
accuracy = float(num_correct) / num_test
print('On a %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))

print('Train data shape: ', data_train.shape)
print('Train labels shape: ', labels_train.shape)
print('Test data shape: ', data_test.shape)
print('Test labels shape: ', labels_test.shape)

"""8- Calculer l'accuracy pour chaque valeurs de k dans un intervalle de 1 à 20."""

k_choices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
k_to_accuracies = {}
for k in k_choices:
    num_test=y_test.shape[0] 
    k_to_accuracies[k] = []
    X_test = data_test
    y_test = labels_test
    X_train = data_train
    y_train = labels_train
    classifier = KNN_model()
    classifier.train(X_train, y_train)
    dists = classifier.compute_distances(X_test)
    y_test_pred = classifier.predict_labels(dists, k)
    num_correct = np.sum(y_test_pred == y_test)
    accuracy = float(num_correct) / num_test
    k_to_accuracies[k].append(accuracy)
print("Ecrire notre accuracy pour différentes valeurs de k:")
print()
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print('k = %d, accuracy = %f' % (k, accuracy))

"""9- Tracer l'évolution de l'accuracy en fonction des k."""

import matplotlib.pyplot as plt
for k in k_choices:
    accuracies = k_to_accuracies[k]
    plt.scatter([k] * len(accuracies), accuracies)

plt.title('Evolution de l accuracy en fonction de k')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()
plt.savefig('knn.png')
plt.close