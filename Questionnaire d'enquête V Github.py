import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.linear_model import Lasso
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import mglearn
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("Observatoirevd_Enquete_2023.csv", sep=';')

# Prepare the y-axis data (All categories are: Sport;Bien-être & Beauté;Patrimoine & Culture;Divertissement;Gastronomie & Vin;Transports touristiques;Nature & Paysage;Autre)
nature_et_paysage=[]
patrimoine_et_culture=[]
gastronomie_et_vin=[]
sport=[]
for line in data['72. ACTIVITES_LOISIRS_NEW']:
    if "Nature & Paysage" in str(line):
        nature_et_paysage.append(1)
    else:
        nature_et_paysage.append(0)
    if "Patrimoine & Culture" in str(line):
        patrimoine_et_culture.append(1)
    else:
        patrimoine_et_culture.append(0)
    if "Gastronomie & Vin" in str(line):
        gastronomie_et_vin.append(1)
    else:
        gastronomie_et_vin.append(0)
    if "Sport" in str(line):
        sport.append(1)
    else:
        sport.append(0)
activities = pd.concat([pd.Series(nature_et_paysage, name='Nature_Paysage'),
                pd.Series(patrimoine_et_culture, name='Patrimoine_Culture'),
                pd.Series(gastronomie_et_vin, name='Gastronomie_Vin'),
                pd.Series(sport, name='Sport')], axis=1)
data=pd.concat([data, activities], axis=1)
data_without_nan = data.dropna()

# Prepare the x-axis data Statut (All categories are: Sport;Bien-être & Beauté;Patrimoine & Culture;Divertissement;Gastronomie & Vin;Transports touristiques;Nature & Paysage;Autre)
avec_des_enfants=[]
avec_des_amis=[]
en_couple=[]
avec_des_parents=[]
seul=[]
en_groupe=[]
for line in data_without_nan['13. STATUT']:
    if "Avec un/des enfant(s)" in str(line):
        avec_des_enfants.append(1)
    else:
        avec_des_enfants.append(0)
    if "Avec un/des ami(s)" in str(line):
        avec_des_amis.append(1)
    else:
        avec_des_amis.append(0)
    if "En couple" in str(line):
        en_couple.append(1)
    else:
        en_couple.append(0)
    if "Avec un/des parent(s)" in str(line):
        avec_des_parents.append(1)
    else:
        avec_des_parents.append(0)
    if "Seul" in str(line):
        seul.append(1)
    else:
        seul.append(0)
    if "En groupe" in str(line):
        en_groupe.append(1)
    else:
        en_groupe.append(0)
visiteurs = pd.concat([pd.Series(avec_des_enfants, name='avec_des_enfants'),
                pd.Series(avec_des_amis, name='avec_des_amis'),
                pd.Series(en_couple, name='en_couple'),
                pd.Series(avec_des_parents, name='avec_des_parents'),
                pd.Series(seul, name='seul'),
                pd.Series(en_groupe, name='en_groupe')], axis=1)

# Create a common index of the two dataframes
data_without_nan['index'] = range(len(data_without_nan))
visiteurs['index'] = range(len(visiteurs))
# Merge the two dataframes using the common index
merged_df = pd.merge(data_without_nan, visiteurs, on='index', how='inner')
# Drop the common index
merged_df.drop(columns=['index'], inplace=True)
visiteurs.drop(columns=['index'], inplace=True)


data_without_nan.to_csv('data_without_nan.csv', encoding='utf-8',sep=';',index=False)

# Switch categorial to numerical variable
def map_fidelite(reponse):
    if reponse == "Non, c'est la première fois":
        return 0
    elif reponse == "Oui, une fois":
        return 1
    elif reponse == "Oui, entre 2 et 10 fois":
        return 2
    elif reponse == "Oui, plus de 10 fois":
        return 3
    else:
        return None
# Apply the function map_fidelite to create a new column
data_without_nan['11. FIDELITE_2_numerique'] = data_without_nan['11. FIDELITE_2'].apply(map_fidelite)
est_suisse = (data_without_nan['31. PAYS'] == "Suisse").astype(int)
a_l_hotel = (data_without_nan['24. HEBERGEMENT'] == "Hôtel").astype(int)


X_quantitative = data_without_nan[['Satisfaction estimée', 'Âge estimé', 'Dépenses par jour catégorie']] #,'11. FIDELITE_2_numerique'
X_categorical = data_without_nan[['7. MOTIFS_SEJOUR', '50. LANG_SAISIE']]

encoder = OneHotEncoder()
# Encode the categorical values into a (scipy) sparse matrix (matrice creuse)
X_categorical_encoded = encoder.fit_transform(X_categorical)
# Convert the result in pandas dataframe
X_categorical_df = pd.DataFrame(X_categorical_encoded.toarray(), columns=encoder.get_feature_names_out())


# Create a common index of the two dataframes
X_categorical_df['index'] = range(len(X_categorical_df))

# Merge the binary variables using a common index
a_l_hotel=pd.DataFrame(a_l_hotel)
a_l_hotel=a_l_hotel.rename(columns={"24. HEBERGEMENT":"a_l_hotel"})
a_l_hotel['index'] = range(len(a_l_hotel))
X_categorical_df = pd.merge(X_categorical_df, a_l_hotel, on='index', how='inner')
est_suisse=pd.DataFrame(est_suisse)
est_suisse=est_suisse.rename(columns={"31. PAYS":"est_suisse"})
est_suisse['index'] = range(len(est_suisse))
X_categorical_df = pd.merge(X_categorical_df, est_suisse, on='index', how='inner')
X_categorical_df.drop(columns=['index'], inplace=True)

# Concatenate the type of visitors to the categorial one-hot encoded variables
X_categorical_df=pd.concat([X_categorical_df, visiteurs],axis=1)

# Scale the quantitative data, so that they have the same inpact on the regression.
standardscaler = StandardScaler() # StandardScaler standardizes the data with a mean of 0 and a standard deviation of 1.
minmaxscaler=MinMaxScaler() # MinMaxScaler normalizes the data, by default between 0 and 1

X_quantitative=pd.DataFrame(X_quantitative)
X_quantitative=X_quantitative.rename(columns={0:"Satisfaction estimée", 1:"Âge estimé", 2:"Dépenses par jour catégorie"})
print ("X_quantitative")
print (X_quantitative)

y = data_without_nan[['Sport']] #, 'Nature_Paysage', 'Patrimoine_Culture', 'Gastronomie_Vin', 'Sport'

X_train_cat, X_test_cat, X_train_quant, X_test_quant, y_train, y_test = train_test_split(X_categorical_df, X_quantitative, y, test_size=0.2, random_state=42)

minmaxscaler.fit(X_train_quant)
X_train_scaled = pd.concat([pd.DataFrame(minmaxscaler.transform(X_train_quant)),X_train_cat.reset_index()], axis=1)
X_train_scaled.drop(columns=['index'], inplace=True)
X_train_scaled.columns = X_train_scaled.columns.astype(str)
X_train_scaled.to_csv('X_train_scaled.csv', encoding='utf-8',sep=';',index=False)

X_test_scaled = pd.DataFrame(minmaxscaler.transform(X_test_quant))

X_train=pd.concat([X_train_quant.reset_index(drop=True),X_train_cat.reset_index(drop=True)],axis=1)
X_test=pd.concat([X_test_quant.reset_index(drop=True),X_test_cat.reset_index(drop=True)],axis=1)
X_train.columns = X_train.columns.astype(str)

X, y = mglearn.datasets.load_extended_boston()

# Lasso regression: when using the lasso, some coefficients are exactly zero.
lasso = Lasso().fit(X_train, y_train)
print("Lasso Default 1.00 Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

lasso002 = Lasso(alpha=0.02).fit(X_train, y_train)
print("Lasso 0.02 Training set score: {:.2f}".format(lasso002.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso002.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso002.coef_ != 0)))

# Decision Trees
simple_tree = DecisionTreeClassifier(max_depth=4, random_state=0) # Setting either max_depth, max_leaf_nodes, or min_samples_leaf is sufficient to prevent overfitting
simple_tree.fit(X_train, y_train)
print("Decision Tree's Accuracy on training set: {:.3f}".format(simple_tree.score(X_train, y_train)))
print("Decision Tree's Accuracy on test set: {:.3f}".format(simple_tree.score(X_test, y_test)))

print("Feature importances:\n{}".format(simple_tree.feature_importances_))

def plot_feature_importances_X_train(model):
    n_features = X_train.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), X_train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

# Gradient boosted regression trees
gbrt = GradientBoostingClassifier(random_state=0, max_depth=1) #To reduce overfitting, we could either apply stronger pre-pruning by limiting the maximum depth (max_depth=1) or lower the learning rate (learning_rate=0.01).
gbrt.fit(X_train, y_train)
print("Gradient boosted regression tree's accuracy on training set: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Gradient boosted regression tree's accuracy on test set: {:.3f}".format(gbrt.score(X_test, y_test)))

# Support Vector Machine
X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
sv = svm.support_vectors_
# Class labels of support vectors are given by the sign of the dual coefficients
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.show()

# Neural Networks (MultiLayer Perceptrons)
mlp = MLPClassifier(random_state=0).fit(X_train.iloc[:,0:2], y_train.iloc[:, 0])
mglearn.plots.plot_2d_separator(mlp, X_train.iloc[:,0:2].values, fill=True, alpha=.3)
mglearn.discrete_scatter(X_train.iloc[:, 0].values, X_train.iloc[:, 1].values, y_train.iloc[:, 0])
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")

# Neural Networks (MultiLayer Perceptrons)
mlp = MLPClassifier(random_state=0).fit(X_train.to_numpy()[:,:2], y_train.to_numpy())
mglearn.plots.plot_2d_separator(mlp, X_train.to_numpy()[:,:2], fill=True, alpha=.3)
mglearn.discrete_scatter(X_train.to_numpy()[:,0], X_train.to_numpy()[:,1], y_train.to_numpy()[:,0])
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
# Accuracy of the neural network
mlp = MLPClassifier(random_state=0, alpha=1).fit(X_train.to_numpy(), y_train.to_numpy())
print("Accuracy on training set: {:.3f}".format(mlp.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(mlp.score(X_test, y_test)))

# Visualisation of the weights for interpretation
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(18), X_train.columns)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()

# Principal Components Analysis
pca = PCA(n_components=2)
pca.fit(X_train_scaled)
# transform data onto the first two principal components
X_pca = pca.transform(X_train_scaled)
print("Original shape: {}".format(str(X_train_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))
plt.figure(figsize=(8, 8))
indices_0 = np.where(y_train == 0)
indices_1 = np.where(y_train == 1)
# Differentiate the values of y_train with blue and red points
plt.scatter(X_pca[indices_0, 0], X_pca[indices_0, 1], color='b', label='Classe 0')
plt.scatter(X_pca[indices_1, 0], X_pca[indices_1, 1], color='r', label='Classe 1')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Data visualisation after ACP')

# K-means classification
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X_train_scaled)
print (type(kmeans))
print("Cluster memberships:\n{}".format(kmeans.labels_[:20]))
print("Predicted cluster memberships:\n{}".format(kmeans.predict(X_train_scaled)[:20]))
mglearn.discrete_scatter(X_train_scaled.iloc[:, 0], X_train_scaled.iloc[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2, 3, 4, 5], markers='^', markeredgewidth=2)
plt.show()
 
# DBSCAN classification (“densitybased spatial clustering of applications with noise”)
dbscan = DBSCAN(eps=1.35, min_samples=20) #parameters: min_samples (2-5), eps (1.0-3.0)
clusters = dbscan.fit_predict(X_train_scaled)
print("Cluster memberships:\n{}".format(clusters[:20]))
mglearn.discrete_scatter(X_train_scaled.iloc[:, 0], X_train_scaled.iloc[:, 1], dbscan.labels_, markers='o')
print("Cluster sizes DBSCAN: {}".format(np.bincount(dbscan.labels_+1)))


# Principal Components Analysis colored according to DBSCAN groups 
pca = PCA(n_components=2)
dbscan = DBSCAN(eps=1.35, min_samples=20) #parameters: min_samples (2-5), eps (1.0-3.0)
clusters = dbscan.fit_predict(X_train_scaled)
pca.fit(X_train_scaled)
# Transform data onto the first two principal components
X_pca = pca.transform(X_train_scaled)
plt.figure(figsize=(8, 8))
# Fetch the indices of the values of y_train equal to x (-1 <= x <= 4)
indices_minus_1 = np.where(dbscan.labels_ == -1)
indices_0 = np.where(dbscan.labels_ == 0)
indices_1 = np.where(dbscan.labels_ == 1)
indices_2 = np.where(dbscan.labels_ == 2)
indices_3 = np.where(dbscan.labels_ == 3)
indices_4 = np.where(dbscan.labels_ == 4)
# Displays the points with different colors for each group
plt.scatter(X_pca[indices_minus_1, 0], X_pca[indices_minus_1, 1], color='b', label='Classe -1')
plt.scatter(X_pca[indices_0, 0], X_pca[indices_0, 1], color='r', label='Classe 0')
plt.scatter(X_pca[indices_1, 0], X_pca[indices_1, 1], color='g', label='Classe 1')
plt.scatter(X_pca[indices_2, 0], X_pca[indices_2, 1], color='y', label='Classe 2')
plt.scatter(X_pca[indices_3, 0], X_pca[indices_3, 1], color='m', label='Classe 3')
plt.scatter(X_pca[indices_4, 0], X_pca[indices_4, 1], color='c', label='Classe 4')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Data visualisation after ACP')

# Principal Components Analysis colored according to k-means groups 
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X_train_scaled)
# Count the number of members of each group
counts = np.bincount(kmeans.labels_)
print ("Number of members of each of the k groups:")
print (counts)
# keep the first two principal components of the data
pca = PCA(n_components=2)
pca.fit(X_train_scaled)
# transform data onto the first two principal components
X_pca = pca.transform(X_train_scaled)
plt.figure(figsize=(8, 8))
# Fetch the indices of the values of y_train equal to x (0 <= x <= 5)
indices_0 = np.where(kmeans.labels_ == 0)
indices_1 = np.where(kmeans.labels_ == 1)
indices_2 = np.where(kmeans.labels_ == 2)
indices_3 = np.where(kmeans.labels_ == 3)
indices_4 = np.where(kmeans.labels_ == 4)
indices_5 = np.where(kmeans.labels_ == 5)
# Displays the points with different colors for each group
plt.scatter(X_pca[indices_0, 0], X_pca[indices_0, 1], color='r', label='Classe 0')
plt.scatter(X_pca[indices_1, 0], X_pca[indices_1, 1], color='g', label='Classe 1')
plt.scatter(X_pca[indices_2, 0], X_pca[indices_2, 1], color='y', label='Classe 2')
plt.scatter(X_pca[indices_3, 0], X_pca[indices_3, 1], color='m', label='Classe 3')
plt.scatter(X_pca[indices_4, 0], X_pca[indices_4, 1], color='c', label='Classe 4')
plt.scatter(X_pca[indices_5, 0], X_pca[indices_5, 1], color='b', label='Classe 5')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Data visualisation after ACP')
