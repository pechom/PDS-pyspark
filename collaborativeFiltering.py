from pyspark import SparkContext
import os
import math
import itertools
from math import sqrt
from operator import add
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
#from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating  -- kedze uz pouzivam dataframe, musim pouzit verziu ALS pre dataframe, nie RDD

# pouzivam sql lebo ALS nepovoluje string ako id (v datasete mam pri hodnoteniach meno a nie ID pouzivatela) a preto musim robit join k datam pouzivatela
sc = SparkContext(master="local",appName="Recommender")
sqlContext = SQLContext(sc)

# nacitanie dataframov
UsersDF = sqlContext.read.format('csv').options(header='true', inferschema='true').load('/home/ryuu/Downloads/myanimelist/users_cleaned.csv')
AnimesDF = sqlContext.read.format('csv').options(header='true', inferschema='true').load('/home/ryuu/Downloads/myanimelist/anime_cleaned.csv')
ScoresDF = sqlContext.read.format('csv').options(header='true', inferschema='true').load('/home/ryuu/Downloads/myanimelist/animelists_cleaned.csv')

# vybratie relevantnych dat
scores_data = ScoresDF.select('username', 'anime_id','my_score')
anime_data = AnimesDF.select('anime_id', 'title','title_english','score','scored_by')
anime_data = anime_data.filter(anime_data.scored_by > 100) # tie ktore maju malo hodnoteni neberiem do uvahy
users_data = UsersDF.select('username', 'user_id', 'stats_mean_score')

# vytvaram jednu tabulku v ktorej su data potrebne na ucenie  
full_data = scores_data.join(users_data, "username").join(anime_data, "anime_id")
recommend_data = full_data.select("user_id", "anime_id", "my_score")

# niektore ciselne polia mi inferschema dalo ako string preto ich musim precastovat
recommend_data = recommend_data .withColumn("anime_id", recommend_data ["anime_id"].cast(IntegerType()))
recommend_data = recommend_data .withColumn("my_score", recommend_data ["my_score"].cast(FloatType()))

# samotne ucenie
training_RDD, test_RDD = recommend_data.randomSplit([0.8, 0.2], seed=42)
test_for_predict_RDD = test_RDD.select("user_id", "anime_id")
als = ALS(maxIter=5, regParam=0.01, userCol="user_id", itemCol="anime_id", ratingCol="my_score",coldStartStrategy="drop")
model = als.fit(training_RDD)
predictions = model.transform(test_RDD) # s tymto by som mohol este zistit chybu

# pre kazdeho pouzivatela urobim predikciu pre 3 filmy ktore by sa mu mali najviac pacit
userRecs = model.recommendForAllUsers(3)
userRecs = userRecs.withColumn("predicted_score", explode(userRecs.recommendations))
userRecs = userRecs.join(users_data, "user_id")
userRecs = userRecs.join(anime_data, userRecs.predicted_score.anime_id == anime_data.anime_id)
userRecs = userRecs.select("user_id", "username", "anime_id", "title", "title_english", "score", "scored_by", "predicted_score")
print(userRecs.take(9))