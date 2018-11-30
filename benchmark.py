import time
import psutil
import pandas as pd
import numpy  as np
import pyspark.sql
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.sql import SQLContext
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession

# Init spark configuration
spark = SparkSession.builder.appName("OLS").getOrCreate()
sc = spark.sparkContext
sqlContext = SQLContext(sc)


def dataParse(dat):
    return (dat[0], Vectors.dense(dat[1:]))


def testOLSSpark(predictor, response):
    # cumulative count initialization
    tot_time = 0
    tot_run = 0

    # Create dictionary to store data
    record = []

    # Pre-process data
    data_raw = pd.concat([response, predictor], axis=1)
    data_tmp = sqlContext.createDataFrame(data_raw)
    data_rdd = data_tmp.rdd.map(dataParse)
    # set partition as 4
    data_reg = sqlContext.createDataFrame(data_rdd, ['label', 'features']).rdd.repartition(4)
    dat_split = data_reg.randomSplit([0.8, 0.2], 88)

    # Set stopping time as 1s for ease, will change to 3600s in final version.
    while (tot_time < 7200):
        benchmark_starttime = time.time()

        # For each run, record time, cpu and I/O
        io_disk_start = psutil.disk_io_counters(perdisk=False)
        io_net_start = psutil.net_io_counters(pernic=False)
        cpu_util_start = psutil.cpu_percent(interval=None)
        mem_used_start = psutil.virtual_memory().used
        algo_starttime = time.time()

        # Pyspark MLlib OLS Regression
        # Parameter setting and model configuration
        ols_model = LinearRegression(standardization=False, elasticNetParam=None)
        cv_model = ols_model.fit(dat_split[0])
        actAndPred = cv_model.transform(dat_split[1])

        algo_timetaken = time.time() - algo_starttime
        mem_used_end = psutil.virtual_memory().used
        cpu_util_end = psutil.cpu_percent(interval=None)
        io_disk_end = psutil.disk_io_counters(perdisk=False)
        io_net_end = psutil.net_io_counters(pernic=False)

        # Calculate Model Accuracy
        y_test_mean = (dat_split[1].rdd.map(lambda p: p.label).reduce(lambda x, y: x + y)) / (dat_split[1].count())
        RSS = actAndPred.rdd.map(lambda p: (p.label - p.prediction) ** 2).reduce(lambda x, y: x + y)
        TSS = actAndPred.rdd.map(lambda p: (p.label - y_test_mean) ** 2).reduce(lambda x, y: x + y)
        RSquared = (TSS - RSS) / TSS

        # Store results
        cur_record = {}
        cur_record['res_time'] = algo_timetaken
        cur_record['accuracy'] = RSquared
        cur_record['cpu_util'] = (cpu_util_end - cpu_util_start) \
            if (cpu_util_end - cpu_util_start) > 0 else 0
        cur_record['mem_use'] = (mem_used_end - mem_used_start) \
            if (mem_used_end - mem_used_start) > 0 else 0
        cur_record['io_disk_rd_count'] = io_disk_end[0] - io_disk_start[0]
        cur_record['io_disk_wt_count'] = io_disk_end[1] - io_disk_start[1]
        cur_record['io_disk_rd_byte'] = io_disk_end[2] - io_disk_start[2]
        cur_record['io_disk_wt_byte'] = io_disk_end[3] - io_disk_start[3]
        cur_record['io_disk_rd_time'] = io_disk_end[4] - io_disk_start[4]
        cur_record['io_disk_wt_time'] = io_disk_end[5] - io_disk_start[5]
        cur_record['io_net_byte_sent'] = io_net_end[0] - io_net_start[0]
        cur_record['io_net_byte_recv'] = io_net_end[1] - io_net_start[1]
        cur_record['io_net_packet_sent'] = io_net_end[2] - io_net_start[2]
        cur_record['io_net_packet_recv'] = io_net_end[3] - io_net_start[3]
        cur_record['io_net_errin'] = io_net_end[4] - io_net_start[4]
        cur_record['io_net_errout'] = io_net_end[5] - io_net_start[5]

        record.append(cur_record)

        benchmark_timetaken = time.time() - benchmark_starttime
        tot_time += benchmark_timetaken

    param_name = ['res_time', 'accuracy', 'cpu_util', 'mem_use', \
                  'io_disk_rd_count', 'io_disk_wt_count', 'io_disk_rd_byte', \
                  'io_disk_wt_byte', 'io_disk_rd_time', 'io_disk_wt_time', \
                  'io_net_byte_sent', 'io_net_byte_recv', 'io_net_packet_sent', \
                  'io_net_packet_recv', 'io_net_errin', 'io_net_errout']

    result = {}

    for i in param_name:
        cur_param_all = [item[i] for item in record]

        result["avg_" + i] = np.mean(cur_param_all)
        result["sd_" + i] = np.std(cur_param_all)
        result["quantile_" + i] = np.percentile(cur_param_all, [0, 25, 50, 75, 100])

    result["algo"] = "OLS"
    result["library"] = "Pyspark ML"
    result['tot_run'] = len(record)
    result["num_pred"] = predictor.shape[1]
    result["num_obs"] = predictor.shape[0]
    # get num of partition
    result['num_partition'] = data_reg.getNumPartitions()

    return (result)


# get measurement data using simulated datasets
obs_size = [1000]
pred = [500]

# move out of loop
sim_predictor = pd.read_csv("./HannaChPredictor_1k.csv", sep=",")
sim_response = pd.read_csv("./HannaChResponse_1k.csv", sep=",")

for i in pred:
    for j in obs_size:
        cur_predictor = sim_predictor.iloc[: j, : i]
        cur_response = sim_response.iloc[:j, 0]
        cur_result = testOLSSpark(cur_predictor, cur_response)
        cur_resultDF = pd.DataFrame(cur_result)
        cur_resultDF.to_csv("./testdat/measurementDataSparkOLS_HCH_obs1000_pred" + str(i) + "_2hr.csv", sep=",",
                            header=True, index=False)
        time.sleep(300)
