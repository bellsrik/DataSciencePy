# libraries to use, this is done in spark 1.5
from pyspark import SparkContext, SparkConf
from pyspark.sql import Row
from pyspark import SQLContext
from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD, LinearRegressionModel
from pyspark.sql.types import StringType
from pyspark.sql.types import DoubleType
from pyspark.mllib.linalg import DenseVector
from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vector
from pyspark.ml.feature import PCA
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.regression import GBTRegressor
from pyspark.sql.types import DoubleType

# setup the spark context since we will run from the shell
conf = SparkConf().setAppName('Brian PySpark Test')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
sc.setLogLevel("ERROR")

# load the data from a csvd file
# we can remove the header by setting a value equal to the first row of the rdd
# and then filter it out.  then we define schema from the rdd and use that to make
# a dataframe
data = sc.textFile("hdfs://ch1-hadoop-ns/user/bcraft/boosted_trees_test_data.csv").map(lambda line: line.split(','))
header = data.first()
data = data.filter(lambda row: row != header)
schema = data.map(lambda x: Row(id = x[0], make = x[1], vdps = x[2], label = x[3]))
df = sqlContext.createDataFrame(schema)

# string indexer for our categorical features
# this indexes each categorical feature and we will
# save them in a data frame that maps the make name to the string
# for persistence purposes
indexer = StringIndexer(inputCol="make", outputCol="makeIDX")
df = indexer.fit(df).transform(df)
make_idx_mappings = df.select('make','makeIDX').distinct().show()

# one hot encoder
# this will convert the indexed strings to sparse one hot vectors
# think of this as dummy feature creation
encoder = OneHotEncoder(inputCol= "makeIDX", outputCol="make_sparse_vect")
df = encoder.transform(df)

# spark models expect to see a feature vector and a prediction column
# so we need to put all our features into a vector, in this case
# the sparse vector and vdp count, we also have to do some
# data type transformations from string to double
df = df.withColumn("vdp_int", df["vdps"].cast("double"))
df = df.withColumn("label_int", df["label"].cast("double"))
assembler = VectorAssembler(inputCols = ["make_sparse_vect", "vdp_int"], outputCol='features')
df = assembler.transform(df)

# make the model
# the step size and iterations is touchy so results might be funky
gbt = GBTRegressor(maxIter=100, maxDepth=30, minInstancesPerNode = 30, featuresCol="features", labelCol="label_int").setStepSize(.5)
linearModel = gbt.fit(df)
df = linearModel.transform(df)
print(df.show())
