## counts the number of crimes for different category.
# analysis
crime_cat = spark.sql("SELECT category, COUNT(IncidntNum) AS Numbers_Crime \
                       FROM sf_crime \
                       GROUP BY category \
                       ORDER BY Numbers_Crime DESC")
display(crime_cat)
crime_cat = crime_cat.toPandas()

# visualization
sb.set(style = "whitegrid")
fig = plt.figure(figsize=(25,20))
x = np.arange(len(crime_cat["category"]))
plt.bar(x, crime_cat["Numbers_Crime"], color = sb.color_palette("PiYG", len(x)), data = crime_cat)
plt.ylabel("number of crimes", fontsize=20)
plt.xticks(x, tuple(crime_cat["category"]), fontsize = 10, rotation = 90)
plt.title('number of crimes for different category', fontsize=20)
display(fig)


## Counts the number of crimes for different district, and visualize your results
# analysis
df_district = spark.sql("SELECT PdDistrict, COUNT(*) as Numbers_Crime \
                         FROM sf_crime \
                         GROUP BY PdDistrict \
                         ORDER BY Numbers_Crime DESC")
display(df_district)
df_district = df_district.toPandas()

# visualization
sb.set(style = "whitegrid")
fig = plt.figure(figsize=(25,10))
x = np.arange(len(df_district["PdDistrict"]))
plt.bar(x, df_district["Numbers_Crime"], color = sb.color_palette("PiYG", len(x)), data = df_district)
plt.ylabel("number of crimes", fontsize=20)
plt.xticks(x, tuple(crime_cat["category"]), fontsize = 20, rotation = 15)
plt.title('number of crimes for different district', fontsize=20)
display(fig)


## Count the number of crimes each "Sunday" at "SF downtown"
# analysis
def in_sf_downtown(x,y):
  point = Point(float(y), float(x))
  return poly.contains(point)
sf_downtown_udf = udf(lambda x, y: in_sf_downtown(x,y), StringType())
df_Q3 = df.select(["IncidntNum", "X", "Y", sf_downtown_udf("X","Y").alias('In_downtown'), to_date("Date_time").alias('Date')])
df_Q3.createOrReplaceTempView("sf_crime_Q3")
df_crime_sunday = spark.sql("SELECT *\
                             FROM sf_crime_Q3 \
                             WHERE WEEKDAY(Date) = 6 and In_downtown = true")
display(df_crime_sunday)

# visualization
# draw the plot according to the coordinates from Google map to make sure the region of the polygon is correct
from pyspark.sql.types import FloatType, BooleanType
from pyspark.sql.functions import udf
from shapely.geometry import Polygon, Point
poly = Polygon([(37.797844, -122.407050), (37.798937, -122.398205), (37.798519, -122.397808), 
              (37.795541, -122.396759), (37.794233, -122.395145), (37.786599, -122.404677),
              (37.797844, -122.407050)])  # this is when I know the order of the coordinates

x,y = poly.exterior.xy
fig = plt.figure(figsize=(4,2))
ax = fig.add_subplot(111)
ax.plot(x, y, color='blue', alpha=0.7, linewidth=1)
ax.set_title('Polygon')
display(fig)

## Analysis the number of crime in each month of 2015, 2016, 2017, 2018.
# analysis
df_crime_month = spark.sql("SELECT YEAR(Date_time) AS Year, MONTH(Date_time) AS Month, COUNT(*) AS Numbers_Crime\
                            FROM sf_crime\
                            GROUP BY MONTH(Date_time), YEAR(Date_time)\
                            ORDER BY YEAR(Date_time), MONTH(Date_time)")
display(df_crime_month)
df_crime_month = df_crime_month.toPandas()
df_crime_month

# visualization
fig = plt.figure(figsize=(25,10))
sb.set(font_scale=2, style = "whitegrid", rc={"lines.linewidth": 2.5})
sb.lineplot(x= "Month", y = "Numbers_Crime", hue="Year", data = df_crime_month)
plt.legend(loc='upper right')
display(fig)


## Analysis the number of crime w.r.t the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15, 2018/10/15.
# analysis
df_crime_same_date = spark.sql("SELECT YEAR(Date_time) AS Year, DATE(Date_time) AS Date, HOUR(Date_time) AS Hour, COUNT(*) AS Numbers_Crime\
                                FROM sf_crime\
                                WHERE DATE(Date_time) in ('2015-12-15','2016-12-15', '2017-12-15', '2018-10-15')\
                                GROUP BY HOUR(Date_time), DATE(Date_time), YEAR(Date_time)\
                                ORDER BY YEAR(Date_time), DATE(Date_time), HOUR(Date_time)")

display(df_crime_same_date)
df_crime_same_date = df_crime_same_date.toPandas()
df_crime_same_date

# visualization
fig = plt.figure(figsize=(30,15))
sb.set(font_scale=3, style = "whitegrid", rc={"lines.linewidth": 2.5})
sb.lineplot(x= "Hour", y = "Numbers_Crime", hue="Date", data = df_crime_same_date)
plt.legend(loc='upper left')
display(fig)

## (1) Step1: Find out the top-3 danger disrict (2) Step2: find out the crime event w.r.t category and time (hour) from the result of step 1
# analysis
df_crime_danger_dist = spark.sql("SELECT PdDistrict, COUNT(*) AS Numbers_Crime \
                                  FROM sf_crime \
                                  GROUP BY PdDistrict \
                                  ORDER BY Numbers_Crime DESC \
                                  LIMIT 3")

display(df_crime_danger_dist)

df_crime_danger_cat = spark.sql("SELECT \
                                    Category, \
                                    HOUR(Date_time) AS Hour, \
                                    COUNT(*) AS Numbers_Crime \
                                 FROM sf_crime \
                                 WHERE PdDistrict IN ('SOUTHERN','MISSION','NORTHERN') \
                                 GROUP BY Category, HOUR(Date_time) \
                                 ORDER BY HOUR(Date_time), Numbers_Crime DESC")

display(df_crime_danger_cat)
df_crime_danger_cat = df_crime_danger_cat.toPandas()
df_crime_danger_cat

# visualization
fig = plt.figure(figsize=(50,20))
sb.set(font_scale=3, style = "whitegrid")
sb.barplot(x= "Hour", y = "Numbers_Crime",  data = df_crime_danger_cat)
plt.legend(loc='upper left')
display(fig)

fig = plt.figure(figsize=(30,40))
sb.set(font_scale=2, style = "whitegrid")
sb.barplot(x= "Numbers_Crime", y = "Category", data = df_crime_danger_cat)
plt.legend(loc='upper left')
plt.xticks(rotation = 0)
display(fig)

## For different category of crime, find the percentage of resulition.
# analysis
df_crime_percent_cat = spark.sql("SELECT Category, COUNT(*) AS Numbers_Crime \
                                  FROM sf_crime \
                                  GROUP BY Category")
df_crime_percent_cat_new = df_crime_percent_cat.toPandas()
sum_crime = df_crime_percent_cat_new["Numbers_Crime"].sum()
df_crime_percent_cat_new = df_crime_percent_cat_new.sort_values(by = ["Numbers_Crime"], ascending = False)
df_crime_percent_cat_new["Crime_Percent"] = df_crime_percent_cat_new["Numbers_Crime"].map(lambda x: str(round((float(x)/sum_crime)*100,2))+" %")
df_crime_percent_cat_new.drop(["Numbers_Crime"], axis = 1)


## Spark ML clustering for spatial data analysis
# analysis
cost = np.zeros(10)
for k in range(2,10):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(assembled.sample(False,0.1, seed=42))
    cost[k] = model.computeCost(assembled) # requires Spark 2.0 or later
# Trains a k-means model.
kmeans = KMeans().setK(6).setSeed(1)
model = kmeans.fit(assembled)
# Make predictions
predictions = model.transform(assembled)
# Evaluate clustering by computing Silhouette score
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))
# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# visualization for selecting k
fig, ax = plt.subplots(1,1, figsize =(10,8))
ax.plot(range(2,10),cost[2:10])
ax.set_xlabel('k')
ax.set_ylabel('cost')
display(fig)

# clustered plot
predictions.select("X","Y","prediction").show(truncate = False)
predictions_pd = predictions.toPandas()
fig = plt.figure(figsize=(10,10))
sb.scatterplot(x = "X", y = "Y", hue = "prediction", data = predictions_pd)
plt.legend(loc='upper left')
display(fig)
