# Recommendation with IBM
In this project, we build recommendation engines using the real data from the IBM Watson Studio platform. There are following 2 data sets:

- User interaction data: which user (anonymous email id) accessed which article (title and article id)
- Content data: information about the article(article name, article description etc).

Below is the glance into the data set

<b>User interaction data</b>

<img src="https://github.com/yukiteb/Data-Science-Nanodegree/blob/master/Recommendation%20with%20IBM/user-interaction-data.PNG" width="600" height="180" title="User interaction data">

<b>Content data</b>

<img src="https://github.com/yukiteb/Data-Science-Nanodegree/blob/master/Recommendation%20with%20IBM/content-data.PNG" width="900" height="180">

## Recommendation engines
We build following 2 recommendation eigines:

- <b>Rank based recommendation engine</b>: This recommendation engines simply recommends the most popular article to any user.
- <b>Collaborative filtering</b>: This recommendation engine is built on "user-item matrix". Each row of user-item matrix represents unique user, and each column of user-item matrix represents unique article id. Whenever an user has interacted with an article (regardless of how many times the user interacted with the specific article), we note 1 in the corresponding user-item matrix, otherwise we put zero. During recommendation, the engine finds the most similar user to a given user based on this user-item matrix and recommends an article.

### Collaborative Filtering
In the collaborative filtering, "similarity" is defined with dot product. In a naive method, for an given user, compute the dot product (using the user-item matrix) against the rest of users. Choose the largest dot product and recommend articles that the user has not seen yet. 

- Instead of arbitrarily choosing when we obtain users who are all the same closeness to a given user - choose the users that have the most total article interactions before choosing those with fewer article interactions.
- Instead of arbitrarily choosing articles from the user where the number of recommended articles starts below m and ends exceeding m, choose articles with the articles with the most total interactions before choosing those with fewer total interactions. This ranking should be what would be obtained from the top_articles function you wrote earlier.


