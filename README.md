# Spark-Data-Clustering-and-Spatial-Data-Analysis-on-San-Francisco-Crime-Data
## Situation
Given crime data from San Francisco between 2003 - 2018 with features: IncidntNum, Date, Time, District, and longitude and latitude. 

Original Data Source: https://data.sfgov.org

## Data visualization link
https://public.tableau.com/profile/yen.chen.chou?fbclid=IwAR0GhlFn45Y866NMGzb2bm08g3MlsOmuo9DxEy3OdqUsZ2qMyPQr__zTw5g#!/vizhome/yenchenchou-2003-2018SanFranciscoCrimeDataVisualization/2003-2018SanFranciscoCrimeDataVisualization

## Prerequisites
Before we start, these packages are required for our analysis:
```Python
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import shapely
import warnings
import os
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType, BooleanType
from pyspark.sql.functions import to_timestamp, concat, lit, udf, upper, to_timestamp, to_date
from shapely.geometry import Polygon, Point
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

```

## Task
Cluster the crime area according to the location data and OLAP.

## Action
1. Load the data in SparkDataFrame. Since the data from 20003 - 2017 and 2018 have different attributes and even different values on the same crime type. So I needed to loaded separately and union the value. 
2. Then I did OLAP through Spark SQL and visualization the OLAP result through Matplotlib and Seaborn. 
3. After that, I identify the crime in the downtown of SF by drawing polygon through define by the longitude and latitude. 
4. Latter on I used K-means clustering to group the high-risk area of SF and use root mean square error versus the number of clusters to measure the relative performance of the clusters.

## Result
I identify the top three dangerous areas through kmeans with 3 clusters and specify time with higher crime incidents and which may potentially decrease 12% of crime incidents. Also, I summarize the most frequent type of crime for the government to adjust their policies on crime.

