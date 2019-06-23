# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 18:39:18 2019

@author: Pierre
"""
##########################################################################
# GSCCompetitorsML1
# Auteur : Pierre Rouarch - Licence GPL 3
# Machine Learning sur un univers de concurrence Partie 1
#####################################################################################

###################################################################
# On démarre ici 
###################################################################
#Chargement des bibliothèques générales utiles
import numpy as np #pour les vecteurs et tableaux notamment
import matplotlib.pyplot as plt  #pour les graphiques
#import scipy as sp  #pour l'analyse statistique
import pandas as pd  #pour les Dataframes ou tableaux de données
import seaborn as sns #graphiques étendues
#import math #notamment pour sqrt()
import os

from urllib.parse import urlparse #pour parser les urls
import nltk # Pour le text mining
# Machine Learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#Matrice de confusion
from sklearn.metrics import confusion_matrix
#pour les scores
from sklearn.metrics import f1_score
#from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import accuracy_score


#pip install google  #pour installer la library de Google Search de Mario Vilas
#https://python-googlesearch.readthedocs.io/en/latest/
import googlesearch  #Scrap serps
#pour randomize pause
import random
import time  #pour calculer le 'temps' de chargement de la page
import gc #pour vider la memoire


print(os.getcwd())  #verif
#mon répertoire sur ma machine - nécessaire quand on fait tourner le programme 
#par morceaux dans Spyder.
#myPath = "C:/Users/Pierre/MyPath"
#os.chdir(myPath) #modification du path
#print(os.getcwd()) #verif


############################################
# Calcul de la somme des tf*idf
#somme des TF*IDF pour chaque colonne de tokens calculé avec TfidfVectorizer
def getSumTFIDFfromDFColumn(myDFColumn) :
    from sklearn.feature_extraction.text import TfidfVectorizer
    corpus = myDFColumn.apply(' '.join)
    vectorizer = TfidfVectorizer(norm=None)
    X = vectorizer.fit_transform(corpus)
    return np.sum(X.toarray(), axis=1)




#####################################################################################
# MODELE "UNIVERS DE CONCURRENCE" - 
# A partir des mots clés récupérés depuis Google Search Consol
# On va récupérer les pages/positions/mots clés grâce à un crawler de SERPs :
# GoogleSearch
#  Code source  : https://github.com/MarioVilas/googlesearch
#  #doc https://python-googlesearch.readthedocs.io/en/latest/
#######################################################################################
#Relecture ############
dfGSC1 = pd.read_json("dfGSC1-MAI.json")
dfGSC1.query
dfGSC1.info() # 514  enregistrements.


len(dfGSC1['query'].unique())  #487 mots clés.
#on sauvegarde les données de base de GSC qu'on ajoutera à la fin
dfGSC1Base = pd.DataFrame(columns=['query', 'page', 'position', 'source']) 
dfGSC1Base['query'] = dfGSC1['query']
dfGSC1Base['page'] = dfGSC1['page']
dfGSC1Base['position'] = dfGSC1['position']
dfGSC1Base['source'] = "GSC"  #source:  Google Search Console 


#tous les mots clés à récupérer : pas de doublons
myQueries = dfGSC1['query'].unique()
myQueries.size #487 mots clés

###########################################################################
# Pour scraper les SERPs de Google
# on utilise la bibliothèque GoogleSearch de Mario Vilas :
# https://python-googlesearch.readthedocs.io/en/latest/
# Attention : Systeme sans proxies ! Cela peut durer compte 
# tenu des pauses.
############################################################################

#dataFrame Scrap des pages de Google
dfScrap = pd.DataFrame(columns=['query', 'page', 'position', 'source'])
len(myQueries)


###############################
i=0
myNum=10
myStart=0
myStop=30
#myTbs= "qdr:m"   #recherche sur le dernier mois. pas utilisé.
#tbs=myTbs,
 #pause assez importante pour ne pas bloquer affiner les valeurs si besoin
myLowPause=15
myHighPause=45

#on boucle (peut durer plusieurs heures - faites cela pendant la nuit :-) !!!)
while i < len(myQueries) :
    myQuery = myQueries[i]
    print("PASSAGE NUMERO :"+str(i))
    print("Query:"+myQuery)
    #on fait varier le user_agent et la pause pour ne pas se faire bloquer
    myPause = random.randint(myLowPause,myHighPause)  #pause assez importante pour ne pas bloquer.
    print("Pause:"+str(myPause))
    myUserAgent =  googlesearch.get_random_user_agent() #modification du user_agent pour ne pas bloquer
    print("UserAgent:"+str(myUserAgent))
    df = pd.DataFrame(columns=['query', 'page', 'position', 'source']) #dataframe de travail
    try :
        urls = googlesearch.search(query=myQuery, tld='fr', lang='fr',  safe='off', 
                                   num=myNum, start=myStart, stop=myStop, domains=None, pause=myPause, 
                                   only_standard=False, extra_params={}, tpe='', user_agent=myUserAgent)
         
        for url in urls :
            #print("URL:"+url)
            df.loc[df.shape[0],'page'] = url
        df['query'] = myQuery  #fill avec la query courante
        df['position'] = df.index.values + 1 #position = index +1
        df['source'] = "Scrap"  #fill avec origine de la donnée
        dfScrap = pd.concat([dfScrap, df], ignore_index=True) #récupère dans l'ensemble des scaps
        time.sleep(myPause) #ajoute une pause
        i += 1 
    except :
        print("ERRREUR LECTURE GOOGLE")
        time.sleep(1200) #ajoute une grande pause si plantage
##############################




dfScrap.info()
dfScrapUnique=dfScrap.drop_duplicates()  #on enlève les evéntuels doublons
dfScrapUnique.info()
#Sauvegarde 
dfScrapUnique.to_csv("dfScrapUnique.csv", sep=";", encoding='utf-8', index=False)  #séparateur ; 
dfScrapUnique.to_json("dfScrapUnique.json") 


#Relecture ############
dfScrapUnique = pd.read_json("dfScrapUnique.json")
dfScrapUnique.query
dfScrapUnique.info() #13801

dfQPPS = pd.concat([dfScrapUnique, dfGSC1Base], ignore_index=True)
#Sauvegarde 
#dfQPPS.to_csv("dfQPPS.csv", sep=";", encoding='utf-8', index=False)  #séparateur ; 
dfQPPS.to_json("dfQPPS.json")


################################################################################
# On redémarre ici pour l'exploration ML
################################################################################
#########################################################################
# Détermination de variables techniques et construites à partir des 
# autres variables
#########################################################################
#Relecture ############
dfQPPS = pd.read_json("dfQPPS.json")
dfQPPS.query
dfQPPS.info() #14315 lignes



#création de la variable à expliquer
#Creation des Groupes
dfQPPS.loc[dfQPPS['position'] <= 10.5, 'group'] = 1
dfQPPS.loc[dfQPPS['position'] > 10.5, 'group'] = 0




#création variables explicatives
#Creation de variables d'url à partir de page
dfQPPS['webSite'] = dfQPPS['page'].apply(lambda x: '{uri.scheme}://{uri.netloc}/'.format(uri=urlparse(x)))
dfQPPS['uriScheme'] = dfQPPS['page'].apply(lambda x: '{uri.scheme}'.format(uri=urlparse(x)))
dfQPPS['uriNetLoc'] = dfQPPS['page'].apply(lambda x: '{uri.netloc}'.format(uri=urlparse(x)))
dfQPPS['uriPath'] = dfQPPS['page'].apply(lambda x: '{uri.path}'.format(uri=urlparse(x)))

#Est-ce que le site est en https ?  ici on a plusieurs sites donc peut être intéressant
dfQPPS.loc[dfQPPS['uriScheme']== 'https', 'isHttps'] = 1
dfQPPS.loc[dfQPPS['uriScheme'] != 'https', 'isHttps'] = 0
dfQPPS.info()

#Pseudo niveau dans l'arborescence calculé au nombre de / -2
dfQPPS['level'] = dfQPPS['page'].str[:-1].str.count('/')-2

############################################
#définition du tokeniser pour séparation des mots
tokenizer = nltk.RegexpTokenizer(r'\w+')  #définition du tokeniser pour séparation des mots

#on va décompter les mots de la requête dans le nom du site et l'url complète
#on vire les accents 
queryNoAccent= dfQPPS['query'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

#on tokenize la requete sans accents
dfQPPS['tokensQueryNoAccent'] = queryNoAccent.apply(tokenizer.tokenize) #séparation des mots pour la reqête

#Page
dfQPPS['lenPage']=dfQPPS['page'].apply(len) #taille de l'url complète en charactères
dfQPPS['tokensPage'] =  dfQPPS['page'].apply(tokenizer.tokenize)   #séparation des mots pour l'url entier
dfQPPS['lenTokensPage']=dfQPPS['tokensPage'].apply(len) #longueur de l'url en mots
#mots de la requete dans   Page
dfQPPS['lenTokensQueryInPage'] = dfQPPS.apply(lambda x : len(set(x['tokensQueryNoAccent']).intersection(x['tokensPage'])),axis=1)
#total des fréquences des mots dans  Page
dfQPPS['lenTokensQueryInPageFrequency'] = dfQPPS.apply(lambda x : x['lenTokensQueryInPage']/x['lenTokensPage'],axis=1)
#SumTFIDF
dfQPPS['sumTFIDFPage'] = getSumTFIDFfromDFColumn(dfQPPS['tokensPage'])
dfQPPS['sumTFIDFPageFrequency'] = dfQPPS.apply(lambda x : x['sumTFIDFPage']/(x['lenTokensPage']+0.01),axis=1) 

#WebSite    
dfQPPS['lenWebSite']=dfQPPS['webSite'].apply(len) #taille de l'url complète en charactères
dfQPPS['tokensWebSite'] =  dfQPPS['webSite'].apply(tokenizer.tokenize)   #séparation des mots pour l'url entier
dfQPPS['lenTokensWebSite']=dfQPPS['tokensWebSite'].apply(len) #longueur de l'url en mots
#mots de la requete dans   WebSite
dfQPPS['lenTokensQueryInWebSite'] = dfQPPS.apply(lambda x : len(set(x['tokensQueryNoAccent']).intersection(x['tokensWebSite'])),axis=1)
#total des fréquences des mots dans   WebSite
dfQPPS['lenTokensQueryInWebSiteFrequency'] = dfQPPS.apply(lambda x : x['lenTokensQueryInWebSite']/x['lenTokensWebSite'],axis=1)
#SumTFIDF
dfQPPS['sumTFIDFWebSite'] = getSumTFIDFfromDFColumn(dfQPPS['tokensWebSite']) 
dfQPPS['sumTFIDFWebSiteFrequency'] = dfQPPS.apply(lambda x : x['sumTFIDFWebSite']/(x['lenTokensWebSite']+0.01),axis=1)    
    
#Path   
dfQPPS['lenPath']=dfQPPS['uriPath'].apply(len) #taille de l'url complète en charactères
dfQPPS['tokensPath'] =  dfQPPS['uriPath'].apply(tokenizer.tokenize)   #séparation des mots pour l'url entier
dfQPPS['lenTokensPath']=dfQPPS['tokensPath'].apply(len) #longueur de l'url en mots
#mots de la requete dans   Path
dfQPPS['lenTokensQueryInPath'] = dfQPPS.apply(lambda x : len(set(x['tokensQueryNoAccent']).intersection(x['tokensPath'])),axis=1)
#total des fréquences des mots dans   Path
#!Risque de division par zero on fait une boucle avec un if
dfQPPS['lenTokensQueryInPathFrequency']=0
for i in range(0, len(dfQPPS)) :
    if dfQPPS.loc[i,'lenTokensPath'] > 0 :
        dfQPPS.loc[i,'lenTokensQueryInPathFrequency'] =dfQPPS.loc[i,'lenTokensQueryInPath']/dfQPPS.loc[i,'lenTokensPath']
#SumTFIDF
dfQPPS['sumTFIDFPath'] = getSumTFIDFfromDFColumn(dfQPPS['tokensPath'])   
dfQPPS['sumTFIDFPathFrequency'] = dfQPPS.apply(lambda x : x['sumTFIDFPath']/(x['lenTokensPath']+0.01),axis=1) 


######################################################################
dfQPPS.info()

####sauvegarde sous un autre nom pour cette étape.
#Sauvegarde 
dfQPPS.to_json("dfQPPS1-MAI.json")  

#on libere de la mémoire
del dfQPPS
gc.collect()






#############################################################################
# Machine Learning - Univers de Concurrence avec variables construite
# sur l'url et les mots clés
#############################################################################
#Relecture ############
dfQPPS1 = pd.read_json("dfQPPS1-MAI.json")
dfQPPS1.query
dfQPPS1.info() # 14315 enregistrements.
#on choisit nos variables explicatives
X =  dfQPPS1[['isHttps', 'level', 
             'lenWebSite',   'lenTokensWebSite',  'lenTokensQueryInWebSiteFrequency' , 'sumTFIDFWebSiteFrequency',             
             'lenPath',   'lenTokensPath',  'lenTokensQueryInPathFrequency' , 'sumTFIDFPathFrequency']]  #variables explicatives
y =  dfQPPS1['group']  #variable à expliquer


#on va scaler
scaler = StandardScaler()
scaler.fit(X)


X_Scaled = pd.DataFrame(scaler.transform(X.values), columns=X.columns, index=X.index)
X_Scaled.info()

X_train, X_test, y_train, y_test = train_test_split(X_Scaled,y, random_state=0)

#Méthode des kNN, recherche du meilleur k

nMax=10
myTrainScore =  np.zeros(shape=nMax)
myTestScore = np.zeros(shape=nMax)
myF1Score = np.zeros(shape=nMax)


for n in range(1,nMax) :
    knn = KNeighborsClassifier(n_neighbors=n) 
    knn.fit(X_train, y_train) 
    myTrainScore[n]=knn.score(X_train,y_train)
    print("Training set score: {:.3f}".format(knn.score(X_train,y_train))) #
    myTestScore[n]=knn.score(X_test,y_test)
    print("Test set score: {:.4f}".format(knn.score(X_test,y_test))) #
    y_pred=knn.predict(X_Scaled)
    print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted')))
    myF1Score[n] = f1_score(y, y_pred, average ='weighted')
    myCM = confusion_matrix(y, y_pred)
    print("Accuracy calculated with CM : {:.4f}".format((myCM[0][0] +  myCM[1][1] ) / y.size))
    print("Accuracy by sklearn : {:.4f}".format(accuracy_score(y, y_pred)))  #idem que précédent


#Graphique train score vs test score vs F1score
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot
sns.lineplot(x=np.arange(1,nMax), y=myTrainScore[1:nMax])
sns.lineplot(x=np.arange(1,nMax), y=myTestScore[1:nMax], color='red')
sns.lineplot(x=np.arange(1,nMax), y=myF1Score[1:nMax], color='yellow')
#fig.suptitle("", fontsize=14, fontweight='bold')
ax.set(xlabel='n neighbors', ylabel='Train (bleu) / Test (rouge) / F1 (jaune)',
       title="La pertinence du modèle sur l'univers de concurrence \n est équivalente à celle pour 1 seul site. \nNous l'enrichirons avec d'autres variables par la suite.")
fig.text(.3,-.06,"Classification Knn - Univers de Concurrence - Position  dans 2 groupes \n vs variables construites en fonction des n voisins", 
         fontsize=9)
#plt.show()
fig.savefig("QPPS1-KNN-Classifier-2goups.png", bbox_inches="tight", dpi=600)


#on choist le premier n_neighbor ou myF1Score est le plus grand
#à vérifier toutefois en regardant la courbe.
indices = np.where(myF1Score == np.amax(myF1Score))
n_neighbor =  indices[0][0]
n_neighbor
knn = KNeighborsClassifier(n_neighbors=n_neighbor) 
knn.fit(X_train, y_train) 
print("N neighbor="+str(n_neighbor))
print("Training set score: {:.3f}".format(knn.score(X_train,y_train))) #
print("Test set score: {:.4f}".format(knn.score(X_test,y_test))) #
y_pred=knn.predict(X_Scaled)
print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted')))
#f1Score retenu pour knn 0.8869  #Précédemment sur le modele interne 0.8944


#par curiosité regardons la distribution des pages dans les groupes 
sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot
sns.countplot(dfQPPS1['group'] , order=reversed(dfQPPS1['group'].value_counts().index))
fig.suptitle('Les groupes sont plus équilibrés que précédemment.', fontsize=14, fontweight='bold')
ax.set(xlabel='groupe', ylabel='Nombre de Pages',
       title="le groupe Top 10 (1) représente la moitié de l'autre groupe.")
fig.text(.2,-.06,"Univers de Concurrence - Distribution des pages/positions dans les 2 groupes.", 
         fontsize=9)
#plt.show()
fig.savefig("QPPS1-Distribution-2goups.png", bbox_inches="tight", dpi=600)
         


#Classification linéaire 1 :   Régression Logistique
#on faire varier C : nverse of regularization strength; must be a positive float. 
#Like in support vector machines, smaller values specify stronger regularization.
#C=1 standard
logreg = LogisticRegression(solver='lbfgs').fit(X_train,y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train,y_train)))  
print("Test set score: {:.3f}".format(logreg.score(X_test,y_test))) 
y_pred=logreg.predict(X_Scaled)
print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted')))
#le F1 Score est très mauvais ici 0.5910 << knn  0.8869

logreg100 = LogisticRegression(C=100, solver='lbfgs').fit(X_train,y_train)
print("Training set score: {:.3f}".format(logreg100.score(X_train,y_train)))  
print("Test set score: {:.3f}".format(logreg100.score(X_test,y_test)))  
y_pred=logreg100.predict(X_Scaled)
print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted')))
#pas mieux   0.5910 << knn 0.8869

logreg1000 = LogisticRegression(C=1000, solver='lbfgs').fit(X_train,y_train)
print("Training set score: {:.3f}".format(logreg1000.score(X_train,y_train)))  
print("Test set score: {:.3f}".format(logreg1000.score(X_test,y_test)))  
y_pred=logreg1000.predict(X_Scaled)
print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted')))
#pas mieux  0.5910 << knn 0.8869

logreg001 = LogisticRegression(C=0.01, solver='lbfgs').fit(X_train,y_train)
print("Training set score: {:.3f}".format(logreg001.score(X_train,y_train)))  
print("Test set score: {:.3f}".format(logreg001.score(X_test,y_test)))  
y_pred=logreg001.predict(X_Scaled)
print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted')))
#moins bien  0.5900 << knn 0.8869

#Classification linéaire 2 :  machine à vecteurs supports linéaire (linear SVC).
LinSVC = LinearSVC(max_iter=10000).fit(X_train,y_train)
print("Training set score: {:.3f}".format(LinSVC.score(X_train,y_train)))  
print("Test set score: {:.3f}".format(LinSVC.score(X_test,y_test)))
y_pred=LinSVC.predict(X_Scaled)
print("F1-Score  : {:.4f}".format(f1_score(y, y_pred, average ='weighted')))  
#pas mieux 0.5908 << knn 0.8869


#######################################################################
# Affichage de l'importance des variables pour logreg
#######################################################################
signed_feature_importance = logreg.coef_[0] #pour afficher le sens 
feature_importance = abs(logreg.coef_[0])  #pous classer par importance
#feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

sns.set()  #paramètres esthétiques ressemble à ggplot par défaut.
fig, ax = plt.subplots()  #un seul plot
ax.barh(pos, signed_feature_importance[sorted_idx], align='center')
ax.set_yticks(pos)
ax.set_yticklabels(np.array(X.columns)[sorted_idx], fontsize=8)
fig.suptitle("L'originalité du nom de domaine n'apporte rien.\n A contrario le fait d'avoir la requête dans le nom de domaine est un avantage. \n La longueur en caractères du nom de domaine importe peu ici.", fontsize=10)
ax.set(xlabel='Importance Relative des variables')
fig.text(.3,-.06,"Régression Logistique - Univers de concurrence \n Importance des variables", 
         fontsize=9)
fig.savefig("QPPS1-Importance-Variables-2goups.png", bbox_inches="tight", dpi=600)
##############################################################



##########################################################################
# MERCI pour votre attention !
##########################################################################
#on reste dans l'IDE
#if __name__ == '__main__':
#  main()






    
