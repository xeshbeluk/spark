import pyspark
from pyspark.sql import SparkSession
import pprint
import json
from pyspark.sql.types import StructType, FloatType, LongType, StringType, StructField
from pyspark.sql import Window
from math import radians, cos, sin, asin, sqrt
from pyspark.sql.functions import lead, udf, struct, col

### haversine distance
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return float(c * r)

def To_numb(x):
  x['PosTime'] = int(x['PosTime'])
  x['FSeen'] = int(x['FSeen'])
  x['Lat'] = float(x['Lat'])
  x['Long'] = float(x['Long'])
  return x

sc = pyspark.SparkContext()

#PACKAGE_EXTENSIONS= ('gs://hadoop-lib/bigquery/bigquery-connector-hadoop2-latest.jar')

bucket = sc._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
project = sc._jsc.hadoopConfiguration().get('fs.gs.project.id')
input_directory = 'gs://{}/hadoop/tmp/bigquerry/pyspark_input'.format(bucket)
output_directory = 'gs://{}/pyspark_demo_output'.format(bucket)

spark = SparkSession \
  .builder \
  .master('yarn') \
  .appName('flights') \
  .getOrCreate()

conf={
    'mapred.bq.project.id':project,
    'mapred.bq.gcs.bucket':bucket,
    'mapred.bq.temp.gcs.path':input_directory,
    'mapred.bq.input.project.id': "osu512",
    'mapred.bq.input.dataset.id': 'Planes',
    'mapred.bq.input.table.id': 'plane_loc',
}

## pull table from big query
table_data = sc.newAPIHadoopRDD(
    'com.google.cloud.hadoop.io.bigquery.JsonTextBigQueryInputFormat',
    'org.apache.hadoop.io.LongWritable',
    'com.google.gson.JsonObject',
    conf = conf)

## convert table to a json like object, turn PosTime and Fseen back into numbers... not sure why they changed
vals = table_data.values()
vals = vals.map(lambda line: json.loads(line))
vals = vals.map(To_numb)

##schema 
schema = StructType([
   StructField('FSeen', LongType(), True),
   StructField("Icao", StringType(), True),
   StructField("Lat", FloatType(), True),
   StructField("Long", FloatType(), True),
   StructField("PosTime", LongType(), True)])

## create a dataframe object
df1 = spark.createDataFrame(vals, schema= schema)


df1.repartition(6) 

## create window by partitioning by Icao and ordering by PosTime, then use lead to get next lat long
window = Window.partitionBy("Icao").orderBy("PosTime").rowsBetween(1,1)
df1=df1.withColumn("Lat2", lead('Lat').over(window))
df1=df1.withColumn("Long2", lead('Long').over(window))
df1 = df1.na.drop()
#pprint.pprint(df1.take(5))
#print(df1.dtypes)

# apply the haversine function to each set of coordinates
haver_udf = udf(haversine, FloatType())
df1 = df1.withColumn('dist', haver_udf('long', 'lat', 'long2', 'lat2'))
#pprint.pprint(df1.take(5))

## sum the distances for each Icao to get distance each plane traveled
df1.createOrReplaceTempView('planes')
top = spark.sql("Select Icao, SUM(dist) as dist FROM planes GROUP BY Icao ORDER BY dist desc LIMIT 10 ")
top = top.rdd.map(tuple)
pprint.pprint(top.collect())

# ### convert the dataframe back to RDD
# dist = df1.rdd.map(list)

# ### apply the haversine equation on each row
# dist = dist.map(lambda x: x+[haversine(x[3],x[2],x[6],x[5])])

# ### create rdd of (Icao, dist)
# dist = dist.map(lambda x: (x[1] , x[7]))

# ### sum each by Icao key, and sort
# dist = dist.reduceByKey(lambda x,y: x+y).sortBy(lambda x: x[1], ascending = False)
# pprint.pprint(dist.take(10))

# ### collect total of all flights
# total = dist.values().reduce(lambda x,y: x+y)
# print(total)

##sum the distances for all planes. 
miles = spark.sql("Select SUM(dist) FROM planes")
pprint.pprint(miles.collect())

## deletes the temporary files
input_path = sc._jvm.org.apache.hadoop.fs.Path(input_directory)
input_path.getFileSystem(sc._jsc.hadoopConfiguration()).delete(input_path, True)

