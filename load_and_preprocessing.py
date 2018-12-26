spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()


## Loading data
# data from 2015 - 2017
df_opt1 = spark.read.format("csv").option("header", "true").load("/FileStore/tables/sf_data_15_17.csv")
df_15_17 = df_opt1.select("IncidntNum", "Category", "PdDistrict", "X", "Y", 
                          concat("Date", lit("-"), "Time").alias('Date_time'))
df_15_17 = df_15_17.select("IncidntNum", "Category", "PdDistrict", "X", "Y", 
                          to_timestamp("Date_time", 'MM/dd/yyyy-HH:mm').alias('Date_time'))

display(df_15_17)

# data from 2018
df_opt1_2 = spark.read.format("csv").option("header", "true").load("/FileStore/tables/Police_Department_Incident_Reports__2018_to_Present.csv")
display(df_opt1_2)

df_18 = df_opt1_2.select(df_opt1_2["Incident Number"].alias("IncidntNum"), 
                         upper(df_opt1_2["Incident Category"]).alias("Category"),
                         upper(df_opt1_2["Police District"]).alias("PdDistrict"),
                         df_opt1_2["Longitude"].alias("X"),
                         df_opt1_2["Latitude"].alias("Y"),
                         to_timestamp("Incident Datetime", 'yyyy/MM/dd h:mm:ss a').alias('Date_time'))

display(df_18)


## Data Cleaning
# Correct data type
df = df.select("IncidntNum", "Category", "PdDistrict", df.X.cast("float"), df.Y.cast("float"), "Date_time")
df.printSchema()

# Check for missing value of coordinates and date
print("Number of rows: ", df.count())
print("Number of NULL in X: ", df.where(df["X"].isNull()).count())
print("Number of NULL in Y: ", df.where(df["Y"].isNull()).count())
print("Number of NULL in Date_time: ", df.where(df["Date_time"].isNull()).count())

# Correct the type errors from Category
df = df.na.replace(["FAMILY OFFENSES", 
                    "FORGERY AND COUNTERFEITING", 
                    "HUMAN TRAFFICKING, COMMERCIAL SEX ACTS",
                    "LARCENY THEFT",
                    "MOTOR VEHICLE THEFT?",
                    "WARRANT",
                    "WEAPONS OFFENCE"], 
                   ["FAMILY OFFENSE", 
                    "FORGERY/COUNTERFEITING", 
                    "HUMAN TRAFFICKING (A), COMMERCIAL SEX ACTS",
                    "LARCENY/THEFT",
                    "MOTOR VEHICLE THEFT",
                    "WARRANT",
                    "WEAPONS OFFENSE"])
display(df.groupBy("Category").count().orderBy("Category"))

# Check for missing value of PdDistrict and type error
display(df.groupBy("PdDistrict").count().orderBy("PdDistrict"))

# Draw the distribution of group by Category
df_X_all = df.groupBy("Category").count().orderBy("count").toPandas()
fig = plt.figure(figsize=(25,12))
sb.set(font_scale=0.8, style = "whitegrid")
sb.barplot(x= "count", y = "Category", data = df_X_all)
plt.legend(loc='upper right')
display(fig)

# Draw the distribution of the NULL coordinates group by Category
# The original distribution group by each category has similar distribution the null's. Also, it is hard to impute the missing value of cooridinates and hence we delete the whole row if there is any missing value.
df_X_miss = df.where(df["X"].isNull()).groupBy("Category").count().orderBy("count").toPandas()
fig = plt.figure(figsize=(15,8))
sb.set(font_scale=0.8, style = "whitegrid")
sb.barplot(x= "count", y = "Category", data = df_X_miss)
plt.legend(loc='upper right')
display(fig)

# Outliers check
Q1_x, Q3_x = df.approxQuantile("X", [0.25, 0.75], 0.0)
x_min_range = Q1_x - (Q3_x - Q1_x)*1.5
x_max_range = Q3_x + (Q3_x - Q1_x)*1.5

Q1_y, Q3_y = df.approxQuantile("Y", [0.25, 0.75], 0.0)
y_min_range = Q1_y - (Q3_y - Q1_y)*1.5
y_max_range = Q3_y + (Q3_y - Q1_y)*1.5

df = df[df["X"] <= max_range]
df = df[df["X"] >= min_range]
df = df[df["Y"] <= max_range]
df = df[df["Y"] >= min_range]

# Drop missing value
df = df.na.drop(how = "any")

# final data
df.createTempView("sf_crime")
