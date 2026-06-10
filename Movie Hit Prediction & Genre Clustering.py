# ------------------------------
# Importing Libraries
# ------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier,plot_tree

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

# ------------------------------
# LOADING DATA
# ------------------------------
df = pd.read_csv("tmdb_5000_movies.csv")

# ------------------------------
# preprocessing/ feature engineering
# ------------------------------

df['genres'] = df['genres'].fillna('')
df['Genres_list'] = df['genres'].str.split()
df['original_language'] = df['original_language'].fillna('Unknown')
df[['budget','revenue','popularity','runtime']] = df[['budget','revenue','popularity','runtime']].fillna(0)

tfidf = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False, max_features=50)
X_genre = tfidf.fit_transform(df['Genres_list']).toarray()

encoder = OneHotEncoder(sparse_output=False)
X_lang = encoder.fit_transform(df[['original_language']])

num_cols = ['budget','revenue','popularity','runtime']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[num_cols])

df["profit"] = df["revenue"] - df["budget"]
df["is_hit"] = (df["profit"] > 0).astype(int)

X_simple = df[['budget','revenue','popularity','runtime','profit']]
X_simple = MinMaxScaler().fit_transform(X_simple)

plt.figure(figsize=(8,6))
corr_matrix = df[['budget','revenue','popularity','runtime','profit']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of Numeric Features")
plt.show()

#Interpretation: first i filled the missing values, then i separated genres so that i can work on them
#while applying models, and used tf-idf to assign related values to similar words. then i used one-hot
#encoding, to assign 0 and 1 values. Scaling was done tou put the data on a range of a scale of 0-1.numeric
#features were taken sepaarte and new features were made to improve model and for classification problem.
#is_hit tells that if a movie is hit-1 or flop-0.
#graph is heatmap showing correlation between features like numeric features hav +ve correlation

# ------------------------------
# TRAIN-TEST SPLIT
# ------------------------------
y_class = df['is_hit']
y_reg = df['popularity']

X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X_simple, y_class, y_reg, test_size=0.25, random_state=42
)

plt.subplot(1,2,1)
plt.hist(y_train_class, bins=2, alpha=0.7, label='Train', color='skyblue', edgecolor='black')
plt.hist(y_test_class, bins=2, alpha=0.7, label='Test', color='orange', edgecolor='black')
plt.xlabel("is_hit (0=Flop, 1=Hit)")
plt.ylabel("Count")
plt.title("Train vs Test Distribution (Classification)")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.hist(y_train_reg, bins=30, alpha=0.7, label='Train', color='skyblue', edgecolor='black')
plt.hist(y_test_reg, bins=30, alpha=0.7, label='Test', color='orange', edgecolor='black')
plt.xlabel("Popularity")
plt.ylabel("Count")
plt.title("Train vs Test Distribution (Regression)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
#Interpretation: to check the working of my models i used training test split to select the train values and test them
#on test values..as i have different problems like supervised and unsupervised and regression etc so i have separated
#them in classification and regression. then i have also used a graph to check both graphs like in classification
#it is showing is hit counts and for regression its showing for popularity in a histogram.

# ------------------------------
# Model improvement
# ------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_simple)

#-------------------------------------------
#k-means
#-------------------------------------------
print("\n---- K-Means: Testing Different k ----")
sil_scores = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    sil_scores[k] = sil

best_k = max(sil_scores, key=sil_scores.get)
print("Best k based on Silhouette Score =", best_k)

print("\n---- Final K-Means Clustering ----")
kmeans = KMeans(n_clusters=best_k, random_state=42)
km_labels = kmeans.fit_predict(X_pca)

print("Silhouette Score:", round(silhouette_score(X_pca, km_labels),3))
print("Calinski-Harabasz Index:", round(calinski_harabasz_score(X_pca, km_labels),3))
print("Davies-Bouldin Index:", round(davies_bouldin_score(X_pca, km_labels),3))

plt.figure(figsize=(7,5))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=km_labels)
plt.title("K-Means Clustering (PCA 2D)")
plt.show()

plt.figure(figsize=(7,5))
plt.bar(['Silhouette'], [silhouette_score(X_pca, km_labels)], color='purple')
plt.ylim(0,1)
plt.title("K-Means Score")

#Interpretation: The resutl of this shows silhoutte score of 66% which is good but not so good, it means
#that clusters are well separated from each other, the harabasz index is so large, which is a good sign
#its higher value shows that clusters are closely related together. the last one is also medium score.
#the best k value is 3. means 3 clusters were formed only and the graph using PCA is clearly showing the d
#difference

# ------------------------------------
# DECISION TREE
# ------------------------------------
print("\n---- Decision Tree with K-Fold ----")

X_tree = df[['budget', 'popularity', 'runtime', 'profit']]
y_tree = df['is_hit']

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    X_tree, y_tree, test_size=0.25, random_state=42
)

dtree_final = DecisionTreeClassifier(random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(dtree_final, X_train_tree, y_train_tree, cv=kf)

print("Mean CV Accuracy:", round(cv_scores.mean(),3))

dtree_final.fit(X_train_tree, y_train_tree)

train_acc = accuracy_score(y_train_tree, dtree_final.predict(X_train_tree))
test_acc = accuracy_score(y_test_tree, dtree_final.predict(X_test_tree))

print("Decision Tree Train Accuracy:", round(train_acc, 3))
print("Decision Tree Test Accuracy:", round(test_acc, 3))

plt.figure(figsize=(20,10))
plot_tree(
    dtree_final,
    feature_names=X_tree.columns,
    class_names=["Flop", "Hit"]
)
plt.title("Decision Tree Structure")
plt.show()


plt.bar(['Train','Test'], [train_acc, test_acc], color='skyblue')
plt.ylim(0,1)
plt.title("Decision Tree Accuracy")
plt.show()
#Interpretation: This model predicts that if the movie is hit or not. the accuracy measure gives 99%
#accuracy which is a perfect result but its not good as the model is traned too much. in next steps its
#overfitting will be reduced for better results. the graph of it is a bar graph showing the accuracy is
#touching the top border showing 100% accuracy.

# -------------------------------
# KNN
# -------------------------------
print("\n---- KNN Classifier ----")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train_class)
knn_pred = knn.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test_class, knn_pred),3))
print("F1 Score:", round(f1_score(y_test_class, knn_pred),3))
print("Confusion Matrix:\n", confusion_matrix(y_test_class, knn_pred))

plt.bar(['KNN'], [accuracy_score(y_test_class, knn_pred)], color='purple')
plt.ylim(0,1)
plt.title("KNN Accuracy")
plt.show()


#Interpretation: knn model gives accuracy upto 88% which is best and normal for my dataset. it gives
#well output but as compared to decision tree its not that good. F1 score is also good which is upto 83%
#showing good score. and then confusion matrix is also showing true positives and negatves etc.
#the graph used is a bar graph, clearly showing how the accuracy is visualized. it see 5 near values and make
#predictions according to it, its not too complex.

# ------------------------------------
# Polynomial Regression
# ------------------------------------
print("\n---- Polynomial Regression ----")

X_features = df[['budget', 'revenue', 'runtime', 'profit']]
y_target = df['popularity']

X_train_feat, X_test_feat, y_train_reg, y_test_reg = train_test_split(
    X_features, y_target, test_size=0.25, random_state=42
)

poly_transformer = PolynomialFeatures(degree=2)
X_train_poly = poly_transformer.fit_transform(X_train_feat)
X_test_poly = poly_transformer.transform(X_test_feat)

model = LinearRegression()
model.fit(X_train_poly, y_train_reg)

y_pred = model.predict(X_test_poly)

mse = mean_squared_error(y_test_reg, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_reg, y_pred)

print("Polynomial Regression MSE:", round(rmse, 3))
print("Polynomial Regression MAE:", round(mae, 3))

plt.scatter(y_test_reg, y_pred, alpha=0.3, color='orange')
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Polynomial Regression (Degree 2)")
plt.show()

plt.subplot(2,3,2)
plt.bar(['RMSE','MAE'], [rmse, mae], color='orange')
plt.title("Polynomial Regression Error")

#Interpretation: i used polynomial regression beacuse valus of my dataset were not good and not
#giving result as a straight line. First i checked with linear regression but accuray was 0.0 which
#mean that model was not working well, then i used this which made the results a little better
#MSE is around 800 which means that errors still there but not too much, MAE is around 14. the result of
#this is not so good. For graph scatter is used which shows wht is the actual popularity against predicted
#popularity.

# ------------------------------------
# ANN
# ------------------------------------
print("\n---- ANN ----")
ann = Sequential()
ann.add(Dense(32,input_shape=(X_train.shape[1],)))
ann.add(Dense(16))
ann.add(Dense(1))

ann.compile(optimizer='adam', loss='mse')
ann.fit(X_train, y_train_reg, epochs=20)

ann_pred = ann.predict(X_test).flatten()
print("ANN MSE:", round(mean_squared_error(y_test_reg, ann_pred),3))
print("ANN MAE:", round(mean_absolute_error(y_test_reg, ann_pred),3))

plt.scatter(y_test_reg, ann_pred, alpha=0.3, color='green')
plt.xlabel("Actual Popularity")
plt.ylabel("ANN Predicted Popularity")
plt.title("ANN Regression Output")
plt.show()

plt.subplot(2,3,3)
plt.bar(['MSE','MAE'], [mean_squared_error(y_test_reg, ann_pred), mean_absolute_error(y_test_reg, ann_pred)], color='green')
plt.title("ANN Error")

#Interpretation: as ANN is for complex paterns and learns. polynomial for my cas was giving results but
#not so good then i applied ANN which is giving better results than polynomial. its MSE is less like around
#700 and the MAE also reduced to 13, it changed but not too much as my dataset has some issues. i used
#scatter graph fo it too like i used in previous, showing comparison of predited and actual. the values shows
#that errors chnaces has been reduced than before.

# ------------------------------------
# Balanced Decision Tree (Overfitting Reduction)
# ------------------------------------
print("\n---- Balanced Decision Tree ----")

X = df[['budget', 'popularity', 'runtime']]
y = df['is_hit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

train_acc, test_acc = [], []

for depth in range(1, 11):
    dtree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dtree.fit(X_train, y_train)

    train_acc.append(dtree.score(X_train, y_train))
    test_acc.append(dtree.score(X_test, y_test))

accuracy_table = pd.DataFrame({
    'Max Depth': range(1, 11),
    'Train Accuracy': train_acc,
    'Test Accuracy': test_acc
})
print(accuracy_table)

plt.plot(range(1, 11), train_acc, marker='o', label='Train Accuracy')
plt.plot(range(1, 11), test_acc, marker='s', label='Test Accuracy')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Balanced Decision Tree")
plt.legend()
plt.grid(True)
plt.show()

#Interpretatrion: before decision tree was giving 99% accuracy almost 100 and thats not okay as model cannot
#predict fully accurate, it was overfit like it learned too much from the datase. to overcome this i tried
#with different depth of the tree with differnt features, like population was not included as it was giving wrong
#results after removing it and setting the depht the accuracy is now 71% which is relatively good for
#model. both train test accuracy were shown by a line graph with values also. I have use hyperparamter
#tuning to reduce its overfitting

# ------------------------------
# Validation on new movie
# ------------------------------
new_movie = {
    'budget': 120_000_000,
    'revenue': 100_000_000,
    'popularity': 85,
    'runtime': 130
}

df_new = pd.DataFrame([new_movie])
df_new['profit'] = df_new['revenue'] - df_new['budget']

X_new = df_new[['budget','popularity','runtime','profit']]

dt_pred_new = dtree_final.predict(X_new)[0]

label = "Hit" if dt_pred_new == 1 else "Flop"
print(f"Decision Tree Prediction for new movie: {label}")


#Interpretation: I did on unseen instance a validation technique like it will compare using budget
#and revenue and check if the movie is hit or flop. profit label was also used and added in the
#dataset. For this it shows that the movie is flop but if we change the values for revenue and
#set it higher than profit the result would be different.

# ------------------------------
# Overall Model Performance Comparison
# ------------------------------

models = ["Decision Tree", "Polynomial Regression", "ANN", "K-Means", "KNN"]
scores = [
    max(test_acc),
    r2_score(y_test_reg, y_pred),
    r2_score(y_test_reg, ann_pred),
    silhouette_score(X_pca, km_labels),
    accuracy_score(y_test_class, knn_pred)
]

plt.figure(figsize=(10,6))
plt.bar(models, scores,
        color=['skyblue','orange','green','purple','red'],
        edgecolor='black')
plt.ylabel("Performance Score")
plt.title("Model Performance Comparison")
plt.ylim(0,1)
plt.grid(axis='y')
plt.xticks(rotation=25)
plt.show()

# ------------------------------
# SUMMARY TABLE
# ------------------------------
print("\n---------------------------------------------------")
print("Summary Table:")
print("---------------------------------------------------\n")

summary_table = pd.DataFrame({
    "Model": [
        "Balanced Decision Tree",
        "Polynomial Regression",
        "Artificial Neural Network (ANN)",
        "K-Means Clustering",
        "KNN Classifier"
    ],
    "Performance Score": [
        round(max(test_acc), 4),
        round(r2_score(y_test_reg, y_pred),4),
        round(r2_score(y_test_reg, ann_pred), 4),
        round(silhouette_score(X_pca, km_labels), 4),
        round(accuracy_score(y_test_class, knn_pred), 4)
    ]
})
print(summary_table)

#Interpretation: summary of my project is that i chose movies data set and applied different models like
#knn,ann,k-means,decision tree and polynomial regression which works for different kinds of problem but the best
#results are given by K-Means as my dataset was suitable ony for clusters problems, i did visualization
#for each step for clarification.
