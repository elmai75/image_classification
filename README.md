# BE1 Apprentissage profond_classification des images
L'objetcif de ce projet est de donner une classification des images du dataset cifar10 qui contient 10 classes(chien,chat,chevale....),pour cela on va utiliser dans ce TD deux  mèthodes, la première méthode c'est Knn, et la deuxième methode qu'on va appliqué c'est le réseaux de neuron profond avec le loss de cross entropy.
Knn est un modèle du machine learning qui prend un nombre k comme paramètre qui signifie le nombre des voisins le plus proche en terme de distance , ensuite il associe au point que l'on veut prédire la classe la plus frequenté dans le nuage  des k points.

## première partie (Knn):
Dans cette partie on va lire le Dataset cifar10 en utilisant la methode unpickle qui retourne un dictionniare de 4 clés b'batch_label', b'labels', b'data', b'filenames', pour notre cas on va utiliser que b'lables' et b'data' pour la définition de la fonction read_cifar et read_cifar_batch. ensuite on va utiliser la bibliothéque sklearn.model_selection pour importer la méthode  train_test_split qui va nous aider faire la sépartaion entre la partie d'entrainement et la partie test.
Ensuite on va définir la fonction qui calcule la distance matricielle entre la partie test et la partie entrainement , l'utilité de la distance euclidienne consiste à définir par suite les k plus proches voisins d'un point en se basant sur cette distance pour identifier ensuite la classe la plus frequenté dans ce nuage de k point.

Aprés la prédiction, il vient la partie de l'evaluation de notre modèle,pour cela , on a utiliser l'accuracy en définisant la fonction evaluate_knn.

L'objectif principale de cette partie c'est de déterminer la valeur optimale de k qui rend l'accuracy maximal.Pour réaliser cette approche on a boucler sur un nombre spécifique de k en calculant à chaque fois l'accuracy pour tracer sa variation en focntion de k

## Deuxième partie(réseaux de neurone):

Dans cette deuxième partie on va essayer de réaliser la classification des images à l'aide de résaux de neuron profond qui contient 3 couche , une couhe d'entrée qui contient 3072 entrée, une couche de sortie qui contient 10 neurons et une couche cahé en utilisant dans les deux dernières couche une fonction d'activation sigmoid.

L'objetcif principale de cette partie c'est d'avoir les meilleurs valeurs de w1,w2,b1,b2 qui minimisent le loss et qui maximise l'accuracy.

En première temps on a utliser le MSE loss qui calcule l'erreur moyen du racine carré de chaque erreur , en bouclant sur un nombre de d'itération bien définie pour faire la mis à jour des poids et des biais à chaque itérations. en utilisant les dérivées partielles de règles de chaines.

Le problème c'est que le Mse Loss n'est pas bien adapté pour traiter les problèmes de classification, c'est pour cela on a utliser l'erreur de cross entroy et en répetant les mêmes démarches .




