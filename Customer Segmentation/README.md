# Customer Segmentation Project
In this project, we apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns.

# File Structures
The files are structured as follows:
```
- finding_donors.ipynb # Jupyter Notebook containing full analysis
- README.md
```

# Results and Analysis
First, we cluster general population of Germany into clusters. We applied the k-means algorithm and elbow method to determine the number of clusters. Below is the average distance of data points as the number of clusters increases.

<img src="https://github.com/yukiteb/Data-Science-Nanodegree/blob/master/Customer%20Segmentation/num_clusters.PNG" width="600" height="400">

From this, we determine that the number of clusters is 10 (this is debatable as the "elbow" is not very clear).

<img src="https://github.com/yukiteb/Data-Science-Nanodegree/blob/master/Customer%20Segmentation/cluster_proportion.PNG" width="600" height="400">


