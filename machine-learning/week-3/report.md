Customer Segmentation and Product Recommendation Engine Using the Customer Personality Analysis Dataset


In this project, I used the Customer Personality Analysis dataset to cluster various customers into 3 broad clusters depending on their characteristics.
The link for the dataset is : "https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data"
 


APPROACH: 

1)Reading and pre-processing the data: Firstly, I Combined expense columns into a single feature: Total_Spending. Similarly, I also combined the kids_home and teens_home columns into asingle one. Then, I deleted the old columns as they were unuseful. ALso, I filled the missing values with the mean of the columns.

2)Applied Feature Engineering and Normalization: I introduced new columns like Customer_Since, Family_Size, Age etc. because these parameters affect the sales in a strong way. Finally, I applied normalization to the numerical columns to speed up the clustering algorithm.

3)I transformed the Features to Make the Data more Gaussian for K-Means to work Efficiently.

4) I applied Elbow Method and Silhouette Scores Method to estimate the number of clusters required. Then I applied K-Means model to the dataset.

5) Then I also estimated the optimum value of the parameter 'epsilon' required for  DBSCAN Clustering and applied this model to the dataset too.

6) Lastly, I made a recommendation system to analyse the purchases of the three clusters and to recommend the products they are most likely to buy. I also classified these clusters into 3 categories of customers - the loyal ones, the thrifty ones and the low frequency ones.


CHALLENGES:

1) The runtime of the Yeo-Johnson algorithm(to make the features more gaussian) was very much and hence I couldn't use it.

2)I faced a lot of difficulties in estimating the correct number of clusters from the Elbow Method as it wasn't clearly indicating a sudden dip in graph. 

3)Moreover, the graph of Silhouette Scores Method was steadily decreasing uptil 7 and hence I couldn't clearly chose a local maxima as the number of clusters.


RESULTS:

1) I concluded that the optimal number of clusters is 3 based on the elbow method and silhouette scores.

2) DBSCAN identified 3 clusters in the data, similar to K-means clustering. The clusters are not well-separated, but DBSCAN was able to identify the outliers effectively.
However, the clusters were not as distinct as in K-means clustering. This was expected as DBSCAN is designed to identify outliers and noise in the data, which may not form well-defined clusters.

3) After analysing the 3 Clusters, I summarized the following for them:

->Cluster 1: Top recommended products for these costumers are Meat Products, Wines, and Gold Products. Moreover, cluster 0 has the lowest average spending and alludes to low frequency buyers.

->Cluster 2: Top recommended products  for these costumers are Wines, Meat Products, and Fish Products. This cluster has the highest average spending and is likely composed of high-value customers.

->Cluster 3: Top recommended products for these costumers are Wines, Meat Products, and Gold Products. This cluster has moderate spending on most product categories.