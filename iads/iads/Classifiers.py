# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2022

# Import de packages externes
from turtle import back
import numpy as np
import pandas as pd
import graphviz as gv
import math
import sys

import random
import copy


def construit_AD_aleatoire(X, Y , epsilon , attr):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = float('-Inf')  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1               # numéro du meilleur attribut (init à -1 (aucun))
        Xbest_seuil = None
                
        Entropie =[]
        Seuil =[]

        tmp = [i for i in range(len(X[0,:]))]

        indices = random.sample(tmp,attr) 

        for i in indices:
            resultat, list_val = discretise(X,Y,i)
            Seuil.append(resultat[0])
            Entropie.append(resultat[1])
            
        i_best = np.argmin(Entropie)
        gain_max = entropie_classe - Entropie[i_best]
        
        Xbest_seuil = Seuil[i_best]
        
        if Xbest_seuil is None:
                Xbest_tuple = ((X,Y),(None,None))
        else:
                Xbest_tuple = partitionne(X,Y,i_best,Xbest_seuil)
        
        ############
        if (gain_max != float('-Inf')):
            noeud = NoeudNumerique(i_best)
            ((left_data,left_class), (right_data,right_class)) = Xbest_tuple
            noeud.ajoute_fils( Xbest_seuil, \
                              construit_AD_aleatoire(left_data,left_class, epsilon, attr), \
                              construit_AD_aleatoire(right_data,right_class, epsilon, attr) )
        else: # aucun attribut n'a pu améliorer le gain d'information
              # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1,"Label")
            noeud.ajoute_feuille(classe_majoritaire(Y))
        
    return noeud


def partitionne(desc_set, label_set, num_col, v) :
    # desc_set and label_set are pretty clear
    # num_col is the colomn we will consider in the partitionning
    # v is the value we will split the set on into two halves
    
    cl_inf = label_set[desc_set[:,num_col]<=v]
    cl_sup = label_set[desc_set[:,num_col]>v]
    desc_inf = desc_set[desc_set[:,num_col]<=v]
    desc_sup = desc_set[desc_set[:,num_col]>v]
    
    # creation of first tuple
    inf_tuple = (desc_inf, cl_inf)
    sup_tuple = (desc_sup, cl_sup)
    
    return inf_tuple, sup_tuple

def tirage(vx, m, r) :
    indices = []
    tmp = vx
    
    if r :
        for i in range(m) :
            indices.append(random.choice(vx))
    else :
        indices = random.sample(vx, m)

    return indices

def echantillonLS(labeledset, m, r) :
    desc_set, label_set = labeledset
    
    L = [i for i in range(len(desc_set))]
    result = tirage(L, m , r)
    return desc_set[result], label_set[result]
            
def discretise(m_desc, m_class, num_col):
    """ input:
            - m_desc : (np.array) matrice des descriptions toutes numériques
            - m_class : (np.array) matrice des classes (correspondant à m_desc)
            - num_col : (int) numéro de colonne de m_desc à considérer
            - nb_classes : (int) nombre initial de labels dans le dataset (défaut: 2)
        output: tuple : ((seuil_trouve, entropie), (liste_coupures,liste_entropies))
            -> seuil_trouve (float): meilleur seuil trouvé
            -> entropie (float): entropie du seuil trouvé (celle qui minimise)
            -> liste_coupures (List[float]): la liste des valeurs seuils qui ont été regardées
            -> liste_entropies (List[float]): la liste des entropies correspondantes aux seuils regardés
            (les 2 listes correspondent et sont donc de même taille)
            REMARQUE: dans le cas où il y a moins de 2 valeurs d'attribut dans m_desc, aucune discrétisation
            n'est possible, on rend donc ((None , +Inf), ([],[])) dans ce cas            
    """
    # Liste triée des valeurs différentes présentes dans m_desc:
    l_valeurs = np.unique(m_desc[:,num_col])
    
    # Si on a moins de 2 valeurs, pas la peine de discrétiser:
    if (len(l_valeurs) < 2):
        return ((None, float('Inf')), ([],[]))
    
    # Initialisation
    best_seuil = None
    best_entropie = float('Inf')
    
    # pour voir ce qui se passe, on va sauver les entropies trouvées et les points de coupures:
    liste_entropies = []
    liste_coupures = []
    
    nb_exemples = len(m_class)
    
    for v in l_valeurs:
        cl_inf = m_class[m_desc[:,num_col]<=v]
        cl_sup = m_class[m_desc[:,num_col]>v]
        nb_inf = len(cl_inf)
        nb_sup = len(cl_sup)
        
        # calcul de l'entropie de la coupure
        val_entropie_inf = entropie(cl_inf) # entropie de l'ensemble des inf
        val_entropie_sup = entropie(cl_sup) # entropie de l'ensemble des sup
        
        val_entropie = (nb_inf / float(nb_exemples)) * val_entropie_inf \
                       + (nb_sup / float(nb_exemples)) * val_entropie_sup
        
        # Ajout de la valeur trouvée pour retourner l'ensemble des entropies trouvées:
        liste_coupures.append(v)
        liste_entropies.append(val_entropie)
        
        # si cette coupure minimise l'entropie, on mémorise ce seuil et son entropie:
        if (best_entropie > val_entropie):
            best_entropie = val_entropie
            best_seuil = v
    
    return (best_seuil, best_entropie), (liste_coupures,liste_entropies)

def construit_AD_num(X,Y,epsilon,LNoms = [], best_carac=None):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    nb_lig, nb_col = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudNumerique(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = float('-Inf')  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1               # numéro du meilleur attribut (init à -1 (aucun))
        Xbest_seuil = None

        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_tuple : le tuple rendu par partionne() pour le meilleur attribut trouvé
        # Xbest_seuil : le seuil de partitionnement associé au meilleur attribut
        #
        # Remarque : attention, la fonction discretise() peut renvoyer un tuple contenant
        # None (pas de partitionnement possible)n dans ce cas, on considèrera que le
        # résultat d'un partitionnement est alors ((X,Y),(None,None))
        
        
        Entropie =[]
        Seuil =[]
        for i in range (len(X[0,:])):
            resultat, list_val = discretise(X,Y,i)
            Seuil.append(resultat[0])
            Entropie.append(resultat[1])
            
        i_best = np.argmin(Entropie)
        if best_carac != None :
            best_carac.append(LNoms[i_best])
        gain_max = entropie_classe - Entropie[i_best]
        
        Xbest_seuil = Seuil[i_best]
        
        if Xbest_seuil is None:
                Xbest_tuple = ((X,Y),(None,None))
        else:
                Xbest_tuple = partitionne(X,Y,i_best,Xbest_seuil)
        
        ############
        if (gain_max != float('-Inf')):
            if len(LNoms)>0:  # si on a des noms de features
                noeud = NoeudNumerique(i_best,LNoms[i_best]) 
            else:
                noeud = NoeudNumerique(i_best)

            ((left_data,left_class), (right_data,right_class)) = Xbest_tuple

            noeud.ajoute_fils( Xbest_seuil, \
                              construit_AD_num(left_data,left_class, epsilon, LNoms,best_carac), \
                              construit_AD_num(right_data,right_class, epsilon, LNoms,best_carac) )
        else: # aucun attribut n'a pu améliorer le gain d'information
              # ARRET : on crée une feuille
            noeud = NoeudNumerique(-1)
            noeud.ajoute_feuille(classe_majoritaire(Y))
        
    return noeud

def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    somme = 0
    
    if 1 in P :
        return somme
    
    for pi in P :
        if pi != 0 :
            somme += pi * math.log(pi)
    return -somme

def entropie(Y):
    
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    probas = [cpt / sum(nb_fois) for cpt in nb_fois]
    return shannon(probas)

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    return valeurs[np.where(nb_fois == max(nb_fois))][0]

def construit_AD(X,Y,epsilon,LNoms = [], best_carac=None):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = float('-Inf')  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1         # numéro du meilleur attribut
        Xbest_valeurs = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur du gain d'information pour chaque attribut.
        
        ##################

        Entropie =[]
        Seuil =[]
        for i in range (len(X[0,:])):
            resultat, list_val = discretise(X,Y,i)
            Seuil.append(resultat[0])
            Entropie.append(resultat[1])
            
        i_best = np.argmin(Entropie)
        gain_max = entropie_classe - Entropie[i_best]
        
        if best_carac != None :
            best_carac.append(LNoms[i_best])

        Xbest_valeurs = np.unique(X[:,i_best])
        

        ##################
        
        #############
        if (gain_max != float('-Inf')) :
            if len(LNoms)>0:  # si on a des noms de features
                noeud = NoeudCategoriel(i_best,LNoms[i_best])    
            else:
                noeud = NoeudCategoriel(i_best)
            for v in Xbest_valeurs:
                noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms,best_carac))

        else: # aucun attribut n'a pu améliorer le gain d'information
                    # ARRET : on crée une feuille
                    noeud = NoeudCategoriel(-1,"Label")
                    noeud.ajoute_feuille(classe_majoritaire(Y))
        
    return noeud

def one_hot_encoder(desc_set) :
    deleted_columns = []
    one_hot_list = []

    # We will use One Hot Encoding to convert any categorial attributes to numerical ones
    for i in range(desc_set.shape[1]) :
        if not isfloat(desc_set[0,i]) : # it's in this case, a categorial column, we apply One Hot Encoding
            deleted_columns.append(i)
            one_hot_list.append(np.array(pd.get_dummies(desc_set[:,i])))

    X = np.delete(desc_set, deleted_columns, 1) # Delete all categorial columns
    
    # Time to merge all the new one hot encoded columns into our new set
    for one_hot in one_hot_list :
        X = np.hstack((X, one_hot))
    
    return X

def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################

class Classifier:
    """ Classe (abstraite) pour représenter un classifieur
        Attention: cette classe est ne doit pas être instanciée.
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        raise NotImplementedError("Please Implement this method")

    def accuracy(self, desc_set, label_set):
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        # ------------------------------
        # COMPLETER CETTE FONCTION ICI :
        
        cpt = 0
        
        for i in range(len(desc_set)) :
            if self.predict(desc_set[i]) == label_set[i] :
                cpt += 1
        
        return cpt/len(desc_set)
        

class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        
        self.input_dimension = input_dimension
        self.k = k
                        
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        
        distances = []
        
        cpt = 0
        
        for i in self.dataset :
            distances.append(np.linalg.norm(x-i))
                
        npdistances = np.argsort(np.asarray(distances))
        
        neighbors = npdistances[0:self.k]
        
        for i in neighbors :
            if self.labelset[i] == 1 :
                cpt += 1
        
        return 2*((cpt / self.k) - 0.5)
            

    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        score = self.score(x)
        
        if score < 0.5 :
            return -1
        else :
            return 1
                
        

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        self.dataset = desc_set
        self.labelset = label_set


class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        
        self.input_dimension = input_dimension
        
        vect_v = []
        
        for i in range(input_dimension) :
            vect_v.append(np.random.uniform(-1,1))
        
        np_v = np.asarray(vect_v)
        self.np_w = np_v / np.linalg.norm(vect_v)
        
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        
        print("Pas d'apprentissage pour ce classifieur\n")
        
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """  
        
        return np.dot(x,self.np_w)

    def predict(self, x):
        
        score = self.score(x)
        
        if score < 0.5 :
            return -1
        else :
            return 1

class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.w = []
        
        if init == 0 :
            self.w = np.zeros(input_dimension)

        elif init == 1 :
            for i in range(0, input_dimension, 2) :
                v = random.uniform(0,1) * 0.001
                self.w.append(v)
                self.w.append((2*v)-1)
                
        
        #raise NotImplementedError("Please Implement this method")
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        indices = [i for i in range(len(desc_set))]
        random.shuffle(indices)

        for i in indices :
            x_i = desc_set[i]
            y_i = label_set[i]
            
            score = self.score(x_i)
            prediction = self.predict(x_i)
            
            if prediction != y_i :
                #correction
                self.w += self.learning_rate * np.dot(y_i, x_i)
        
        
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        
        difference = seuil + 1
        variations = []
        
        for i in range(niter_max) :

            if difference > seuil :
                ancien_w = self.w.copy()
                self.train_step(desc_set, label_set)
                difference = np.linalg.norm(self.w - ancien_w)
                variations.append(difference)
                
            else :
                break
            
        return variations
                
                
        
        
        #raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        
        return np.dot(x,self.w)

        #raise NotImplementedError("Please Implement this method")
        
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0.0 :
            return -1
        else : 
            return 1


class ClassifierPerceptronKernel(Classifier):
    """ Perceptron de Rosenblatt kernelisé
    """
    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : epsilon
                - noyau : Kernel à utiliser
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.noyau = noyau
        self.w = []
        
        if init == 0 :
            self.w = np.zeros(self.noyau.get_output_dim())

        elif init == 1 :
            for i in range(0, self.noyau.get_output_dim(), 2) :
                v = random.uniform(0,1) * 0.001
                self.w.append(v)
                self.w.append((2*v)-1)
        
        self.w = np.reshape(self.w, (self.noyau.get_output_dim(),1))

        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        
        indices = [i for i in range(len(desc_set))]
        random.shuffle(indices)
        
        for i in indices :
            x_i = desc_set[i].copy()
            x_i2 = np.reshape(x_i.copy(),(1,self.input_dimension))
            x_i2 = self.noyau.transform(x_i2)
            y_i = label_set[i]
            
            score = self.score(x_i)
            prediction = self.predict(x_i)
            
            if prediction != y_i :
                #correction
                self.w += (self.learning_rate * np.dot(y_i, x_i2)).T
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        

        difference = seuil + 1
        variations = []
        
        for i in range(niter_max) :
            if difference > seuil :
                ancien_w = self.w.copy()
                self.train_step(desc_set, label_set)
                difference = np.linalg.norm(self.w - ancien_w)
                variations.append(difference)
                
            else :
                break
            
        return variations
        
    def score(self,x):
        """ rend le score de prédiction sur x 
            x: une description (dans l'espace originel)
        """
        x = np.reshape(x.copy(),(1,self.input_dimension))
        x = self.noyau.transform(x)
        return np.dot(x,self.w)

    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description (dans l'espace originel)
        """
        if self.score(x) < 0.0 :
            return -1
        else : 
            return 1
            

class ClassifierArbreNumerique(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set, best_carac=None):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD_num(desc_set,label_set,self.epsilon,self.LNoms, best_carac)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)


class ClassifierPerceptronBiais(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.w = []
        self.allw = []
        
        if init == 0 :
            self.w = np.zeros(input_dimension)

        elif init == 1 :
            for i in range(0, input_dimension, 2) :
                v = random.uniform(0,1) * 0.001
                self.w.append(v)
                self.w.append((2*v)-1)
        
        self.allw.append(self.w.copy())
        
        #raise NotImplementedError("Please Implement this method")
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        indices = [i for i in range(len(desc_set))]
        random.shuffle(indices)
        
        for i in indices :
            x_i = desc_set[i]
            y_i = label_set[i]
            
            score = self.score(x_i)
            prediction = self.predict(x_i)
            
            if score * y_i < 1 :
                #correction
                self.w += self.learning_rate * (np.dot((y_i - score),x_i))
                self.allw.append(copy.deepcopy(self.w))

        
        
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        
        difference = seuil + 1
        variations = []
        
        for i in range(niter_max) :
            if difference > seuil :
                ancien_w = copy.deepcopy(self.w)
                self.train_step(desc_set, label_set)
                difference = np.linalg.norm(self.w - ancien_w)
                variations.append(difference)                
            else :
                break
            
        return variations
                
                
        
        
        #raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        
        return np.dot(x,self.w)

        #raise NotImplementedError("Please Implement this method")
        
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
            
        if self.score(x) < 0.0 :
            return -1
        else : 
            return 1

    def get_allw(self) :
        allw_np = np.asarray(self.allw)
        return allw_np
    

class Perceptron_MC(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, num_of_classes, ClBinaire ,init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.ClBinaire = ClBinaire
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.num_of_classes = num_of_classes
        self.w = []
        self.Classifiers = []
        self.scores = []
        self.Ytmp = []
        
        #raise NotImplementedError("Please Implement this method")
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        
        
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        
        
        for i in range(self.num_of_classes) :
            self.Classifiers.append(copy.deepcopy(self.ClBinaire))
            self.Ytmp.append(np.where(label_set == i,1,-1))
            self.Classifiers[i].train(desc_set,self.Ytmp[i])

        
        #raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        
        scr = []
        
        for c in self.Classifiers : 
            scr.append(c.score(x))
        
        return scr
        #raise NotImplementedError("Please Implement this method")
        
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        
        scores = self.score(x)
        return scores.index(max(scores))



class ClassifierMultiOAA(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, ClBinaire ,init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.ClBinaire = ClBinaire
        self.input_dimension = input_dimension
        self.w = []
        self.Classifiers = []
        self.scores = []
        self.Ytmp = []
        
        #raise NotImplementedError("Please Implement this method")
        
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """
        
        
    def train(self, desc_set, label_set, niter_max=100, seuil=0.01):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """
        
        self.num_of_classes = len(np.unique(label_set))
        
        for i in range(self.num_of_classes) :
            self.Classifiers.append(copy.deepcopy(self.ClBinaire))
            self.Ytmp.append(np.where(label_set == i,1,-1))
            self.Classifiers[i].train(desc_set,self.Ytmp[i])

        
        #raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        
        scr = []
        
        for c in self.Classifiers : 
            scr.append(c.score(x))
        
        return scr
            
        #raise NotImplementedError("Please Implement this method")
        
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        
        scores = self.score(x)
        return scores.index(max(scores))


class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.input_dimension = input_dimension
        self.learning_rate = learning_rate
        self.history = history
        self.niter_max = niter_max
        self.allw = []
        self.w = []
        for i in range(0, input_dimension, 2) :
                v = random.uniform(0,1) * 0.001
                self.w.append(v)
                self.w.append((2*v)-1)
        self.allw.append(self.w)

        #raise NotImplementedError("Please Implement this method")
        
    def train(self, desc_set, label_set, seuil=0.01):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """

        backup_w = np.asarray(self.w) + 1
        j = 0
        
        indices = [i for i in range(len(desc_set))]
        random.shuffle(indices)
        

        while(np.linalg.norm(backup_w - np.asarray(self.w)) > seuil and j < self.niter_max) :

            backup_w = np.asarray(self.w.copy())

            for i in range(len(indices)) :
                x_i = desc_set[indices[i]]
                y_i = label_set[indices[i]]
                score = self.score(x_i)
                gradient = np.dot(x_i.T, (np.dot(x_i,self.w) - y_i))
                self.w -= self.learning_rate * gradient

                if (self.history == True) :
                    self.allw.append(self.w.copy())
            
            j += 1

        print("Convergence en ", j , " iterations")
        
        #raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
        #raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0.0 :
            return -1
        else : 
            return 1


            
        
        

        
        
        
        #raise NotImplementedError("Please Implement this method")
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
        #raise NotImplementedError("Please Implement this method")
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        if self.score(x) < 0.0 :
            return -1
        else : 
            return 1

class ClassifierArbreNumeriqueAlea(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision numérique
    """
    def __init__(self, input_dimension, epsilon, attr):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.attr = attr
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.racine = construit_AD_aleatoire(desc_set,label_set,self.epsilon, self.attr)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        return self.racine.classifie(x)

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)

class ClassifierRandomForest(Classifier) :
    def __init__(self, B, pourcentage, entropie, r, attr):
        self.B = B
        self.pourcentage = pourcentage
        self.entropie = entropie
        self.r = r
        self.attr = attr
        self.arbres = set()
        
    def train(self, desc_set, label_set) :
        self.classes = np.unique(label_set)
        m = int(self.pourcentage * len(desc_set))
        
        for i in range(self.B) :
            ds, ls = echantillonLS((desc_set, label_set), m, self.r)
            print(desc_set.shape[1])
            arbre = ClassifierArbreNumeriqueAlea(desc_set.shape[1], self.entropie, self.attr)
            arbre.train(ds, ls)
            self.arbres.add(arbre)

    def predict(self, exemple) :
        tmp = dict.fromkeys(self.classes,0)
        
        for arbre in self.arbres :
            tmp[arbre.predict(exemple)] += 1
        
        return max(tmp, key = tmp.get) 
        

class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            # print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g


class NoeudNumerique:
    """ Classe pour représenter des noeuds numériques d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.seuil = None          # seuil de coupure pour ce noeud
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, val_seuil, fils_inf, fils_sup):
        """ val_seuil : valeur du seuil de coupure
            fils_inf : fils à atteindre pour les valeurs inférieures ou égales à seuil
            fils_sup : fils à atteindre pour les valeurs supérieures à seuil
        """
        if self.Les_fils == None:
            self.Les_fils = dict()            
        self.seuil = val_seuil
        self.Les_fils['inf'] = fils_inf
        self.Les_fils['sup'] = fils_sup        
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        #############
        
        if self.est_feuille() :
            return self.classe
        
        if exemple[self.attribut] <= self.seuil:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils['inf'].classifie(exemple)
        return self.Les_fils['sup'].classifie(exemple)
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            # print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
        
        #############
        
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc 
            pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, str(self.nom_attribut))
            self.Les_fils['inf'].to_graph(g,prefixe+"g")
            self.Les_fils['sup'].to_graph(g,prefixe+"d")
            g.edge(prefixe,prefixe+"g", '<='+ str(self.seuil))
            g.edge(prefixe,prefixe+"d", '>'+ str(self.seuil))                
        return g

class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set,best_carac=None):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        ##################
        self.racine = construit_AD(desc_set,label_set,self.epsilon,self.LNoms,best_carac)
        ##################
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        ##################
        return self.racine.classifie(x)
        ##################

    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)

class ClassifierBaggingTree(Classifier):

    ###### A COMPLETER
    def __init__(self, B, pourcentage, entropie, r):
        self.B = B
        self.pourcentage = pourcentage
        self.entropie = entropie
        self.r = r
        self.arbres = set()
    
    def train(self, labeledset) :
        
        desc_set, label_set = labeledset
        self.classes = np.unique(label_set)
        
        m = int(self.pourcentage * len(desc_set))
        for i in range(self.B) :
            ds, ls = echantillonLS(labeledset, m, self.r)
            arbre = ClassifierArbreNumerique(len(desc_set[0]), self.entropie)
            arbre.train(ds, ls)
            self.arbres.add(arbre)
    
    
    def predict(self, exemple) :
        tmp = dict.fromkeys(self.classes,0)
        
        for arbre in self.arbres :
            tmp[arbre.predict(exemple)] += 1
        
        return max(tmp, key = tmp.get) 
                      

