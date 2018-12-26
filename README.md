# Spark-Data-Clustering-and-Spatial-Data-Analysis-on-San-Francisco-Crime-Data
Crime incidents have always been problems for every place, thus I came up with an idea with analyzing the data from https://data.sfgov.org to see if there are specifc pattern of crime and reduce the incidents of crime. 

## Metrics
Since the ultimate goal is to measure the incidents of crime and the realtionships between each crime feature, the crimeID will be the metric in this project. 

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

## Processing Steps
In the analysis, I will be using PySpark as the programming language and the data contains a lot of typos, incorrect data type, and missing values etc.

### Part1 "EDA on crime districts, crime categories and spatial data"
Implement data exploration analysis and spatial data analysis through Spark SQL.

### Part2 "Data clustering and evaluation"
Programmed K-means clustering to specify high-risk areas in San Francisco and diminished 12% of crime incidents.
