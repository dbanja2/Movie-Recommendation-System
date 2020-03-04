# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:57:32 2020

@author: dbanja2
"""

#author Dipendra Banjara at lsu dot edu
## ---(Thu Feb 27 13:03:23 2020)---

import pandas as pd
import numpy as np
# data from MovieLens|GroupLens (http://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
movie_data = pd.read_csv('C:/Users/dbanja2/Downloads/ml-latest-small/movies.csv',error_bad_lines=False, header=0, usecols=[0,1,2] ,index_col=0, names=['movieId', 'Title','Genre'])
movie_ratings = pd.read_csv('C:/Users/dbanja2/Downloads/ml-latest-small/ratings.csv', error_bad_lines=False, usecols=[0,1,2],header=0, names=['userId','movieId', 'rating'] )
#movie_ratings.head()
#movie_data.head()
#movie_data.describe()
movieIdNum = len(movie_data) #returns total number of rows
print('The total movies are: ', movieIdNum +1)

#number of rating given by each user
ratingsPerUser = pd.DataFrame({'num of ratings': movie_ratings.groupby('userId')['rating'].count() })
#ratingsPerUser = pd.DataFrame({'num of ratings': movie_ratings.groupby('userId')['rating'].value_counts()})
ratingsPerUser.head(10)

#number of rating given for each movie
ratingsPerMovie = pd.DataFrame({'num of ratings': movie_ratings.groupby('movieId')['rating'].count()})
#ratingsPerMovie = pd.DataFrame({'num of ratings': movie_ratings.groupby('movieId')['rating'].value_counts()})
ratingsPerMovie.head(10)




def movieTitle(movieId):
    title = movie_data.at[movieId, 'Title'] #returns movie title at the movieId
    return title
print(movieTitle(1))


def movieGenre(movieId):  
    genre = movie_data.at[movieId, 'Genre']
    return  genre
print(movieGenre(1))

#for i in range(1, movieIdNum):
#    print(movieTitle(i))
#    print(movieGenre(i))

# Data Preprocessing for huge dataset (However here Not Required)

print('Give the ID of user1:')
user1 = int(input())
print('Give the ID of user2:')
user2 = int(input())

# to select only those movies whose id is present in movie_data
# movie_ratings = movie_ratings[movie_ratings['movieId'].isin(movie_data.index)]
def favMovie(userId, N):
    userRatings = movie_ratings[movie_ratings.userId==userId]
    sortedRatings = pd.DataFrame.sort_values(userRatings,['rating'] ,ascending=False)[:N]
    sortedRatings['Title'] = sortedRatings['movieId'].apply(movieTitle)
    sortedRatings['Genre'] = sortedRatings['movieId'].apply(movieGenre)
    return sortedRatings
#favMovie(1, 10)    
for userId in [user1, user2]:
    N = 5
    print('The', N,'-favorite movies', 'of user ', userId, 'is: \n', favMovie(userId, N))

#Setup Rating Matrix
movie_ratings.shape, movie_data.shape
userPerMovieID = movie_ratings.movieId.value_counts() #returns count of unique values
print(userPerMovieID.head())
print(userPerMovieID.shape)

## Data Preprocessing to obtain less sparse matrix for huge dataset(However here Not Required)
## Take only those movies which are seen by more than 10 users
#movie_ratings = movie_ratings[movie_ratings.index.isin(userPerMovieID[userPerMovieID > 10].index)]
#movie_ratings.shape

userMovieRatingMatrix = pd.pivot_table(movie_ratings, index=['userId'],columns=['movieId'] ,values='rating', margins=True)
print('The rating matrix for each user is:')
print(userMovieRatingMatrix.head(5))


###Find K nearest neighbours
##########------------###########
from scipy.spatial.distance import hamming
# hamming() returns a value which shows the pecentage of disagreement
##https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.signal.hamming.html


# Wrapping it up in a function
def distance(user1, user2):
    try:
        user1_ratings = userMovieRatingMatrix.transpose()[user1]
        user2_ratings = userMovieRatingMatrix.transpose()[user2]
        distance = hamming(user1_ratings, user2_ratings) #hamming distance used for categorical variables
    except:
        distance = np.nan
    return distance
print('The distance between', user1, 'and ', user2, 'is ', distance(user1,user2))

#print(hamming(user1_ratings, user1_ratings))

# Wrapping it up in a function
def nearestNeighbours(user, K):
    allusers = pd.DataFrame(userMovieRatingMatrix.index)
    allusers = allusers[allusers.userId != user] ##Removing the active user
    # calculating distanc of other user with active user and sorting nearest first 
    allusers['distance'] = allusers['userId'].apply(lambda x: distance(user, x))
    KNearestUsers = allusers.sort_values(['distance'], ascending=True)['userId'][:K]
    return KNearestUsers

#KNearestNeighbours = nearestNeighbours(2, 25)
print('Give the number of neearest neighbor "K": \n')
K = int(input())
for user in [user1, user2]:
    KNearestNeighbours = nearestNeighbours(user,K)
    KNearestNeighbours
    print('The', K,'-nearest Neighbour', 'of', user, 'is: \n', KNearestNeighbours)
    

####-----Find Top N Recommendations----####
# Nearest Neighbours ratings

# Wrapping it up in a function
def topN(user,N):
    KnearestUsers = nearestNeighbours(user, K)
    # Nearest Neighbours ratings
    NNRatings = userMovieRatingMatrix[userMovieRatingMatrix.index.isin(KnearestUsers)]
    #Getting the average rating of each movie seen by Nearest Neighbours of active user
    # warning where the columns of NNratings are completely empty(nan)
    avgRating = NNRatings.apply(np.nanmean).dropna()
    # Removing the movies which are already seen by user
    moviesAlreadySeen = userMovieRatingMatrix.transpose()[user].dropna().index
    avgRating = avgRating[~avgRating.index.isin(moviesAlreadySeen)]
    topNMovieId = avgRating.sort_values(ascending=False).index[:N]
    topNMovies = pd.DataFrame({'Movie':pd.Series(topNMovieId).apply(movieTitle), 'Genre':pd.Series(topNMovieId).apply(movieGenre)})
    return topNMovies
    
    
userTopNMovies = topN(user,N)


#print(topN(3,10))

for user in [user1, user2]:
    N = 5
    userTopNMovies = topN(user,N)
    print('The', N,'-top movies recommended for user ', user, 'are: \n', topN(user,N))
    
#favMovie(3,5)

# To remove the RunTimeWarning error 
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

###---For Ploting-----###
import matplotlib.pyplot as plt 
#import seaborn as sns  
#sns.set_style('white') 
#%matplotlib inline 
# plot graph of 'num of ratings column' 
plt.figure(figsize =(5, 4))  
ratingsPerUser['num of ratings'].hist(bins = 3000, facecolor='g')

plt.xlabel('userId')
plt.ylabel('ratings count')
plt.title('Histogram of ratings per user')
#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.xlim(20, 250)
plt.ylim(0, 15)
plt.grid(True)
plt.show()


fig, ax = plt.subplots()
x = ratingsPerUser.index
y = ratingsPerUser['num of ratings']
ax.plot(x ,y, 'bo')
ax.set(xlabel='userId', ylabel='count of ratings', title='ratings per User')
plt.show()
