import time
import psutil
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score
#from sklearn.model_selection import train_test_split

#Write function for benchmarking
def testlm(data, labels, clusters):
    #cumulative count initialization
    tot_time  = 0
    tot_run   = 0

    #Split train and test data
    #predictorTrain, predictorTest, responseTrain, responseTest = \
    #train_test_split(predictor, response, test_size=0.2, random_state=1)
    
    #Create dictionary to store data
    record = []
    
    #Set stopping time as 3s for ease, will change to 3600s in final version.
    while(tot_time < 3600):
        benchmark_starttime = time.time()
        
        #For each run, record time, cpu and I/O
        io_disk_start  = psutil.disk_io_counters(perdisk = False)
        io_net_start   = psutil.net_io_counters(pernic = False)
        cpu_util_start = psutil.cpu_percent(interval = None)
        mem_used_start = psutil.virtual_memory().used
        algo_starttime      = time.time()
        
        km = KMeans(n_clusters = clusters)
        modkm = km.fit(data)

        algo_timetaken = time.time() - algo_starttime
        mem_used_end   = psutil.virtual_memory().used
        cpu_util_end   = psutil.cpu_percent(interval = None)
        io_disk_end    = psutil.disk_io_counters(perdisk = False)
        io_net_end     = psutil.net_io_counters(pernic = False)
        
        #Store results
        cur_record = {}
        cur_record['res_time']           = algo_timetaken
        cur_record['accuracy']           = homogeneity_score(labels,modkm.labels_)
        cur_record['cpu_util']           = (cpu_util_end -cpu_util_start) \
                                            if (cpu_util_end - cpu_util_start) > 0 else 0
        cur_record['mem_use']            = (mem_used_end - mem_used_start) \
                                            if (mem_used_end - mem_used_start) > 0 else 0
        cur_record['io_disk_rd_count']   = io_disk_end[0] - io_disk_start[0]
        cur_record['io_disk_wt_count']   = io_disk_end[1] - io_disk_start[1]
        cur_record['io_disk_rd_byte']    = io_disk_end[2] - io_disk_start[2]
        cur_record['io_disk_wt_byte']    = io_disk_end[3] - io_disk_start[3]
        cur_record['io_disk_rd_time']    = io_disk_end[4] - io_disk_start[4]
        cur_record['io_disk_wt_time']    = io_disk_end[5] - io_disk_start[5]
        cur_record['io_net_byte_sent']   = io_net_end[0]  - io_net_start[0]
        cur_record['io_net_byte_recv']   = io_net_end[1]  - io_net_start[1]
        cur_record['io_net_packet_sent'] = io_net_end[2]  - io_net_start[2]
        cur_record['io_net_packet_recv'] = io_net_end[3]  - io_net_start[3]
        cur_record['io_net_errin']       = io_net_end[4]  - io_net_start[4]
        cur_record['io_net_errout']      = io_net_end[5]  - io_net_start[5]
                
        record.append(cur_record)
        
        benchmark_timetaken = time.time() - benchmark_starttime
        tot_time += benchmark_timetaken
           
    param_name = ['res_time', 'accuracy', 'cpu_util', 'mem_use', \
                  'io_disk_rd_count', 'io_disk_wt_count', 'io_disk_rd_byte' , \
                  'io_disk_wt_byte' , 'io_disk_rd_time' , 'io_disk_wt_time' , \
                  'io_net_byte_sent', 'io_net_byte_recv', 'io_net_packet_sent', \
                  'io_net_packet_recv', 'io_net_errin', 'io_net_errout']
    
    result = {}
    
    for i in param_name:
        cur_param_all = [item[i] for item in record]
        
        result["avg_" + i ]     = np.mean(cur_param_all)
        result["sd_" + i]       = np.std(cur_param_all)
        result["quantile_" + i] = np.percentile(cur_param_all, [0, 25, 50, 75, 100])
    
    result["algo"]    = "KMeans"
    result["library"] = "Python Sklearn"
    result['tot_run'] = len(record)
    result["num_pred"]= len(data.columns)
    result["num_obs"] = len(data)
            
    return (result)

datasets = {
    "./KMeans_datasets/Selected/Cluster_3cl_X2_5K_9K_18229_XID.csv" : 3,
    "./KMeans_datasets/Selected/Cluster_3cl_X10_50K_90K_249040_XID.csv" : 3,
    "./KMeans_datasets/Selected/Cluster_6cl_X4_25K_45K_239335_XID.csv" : 6,
    "./KMeans_datasets/Selected/Cluster_8cl_X2_5K_9K_58322_XID.csv" : 8,
    "./KMeans_datasets/Selected/Cluster_8cl_X3_50K_90K_584499_XID.csv" : 8,
    "./KMeans_datasets/Selected/Cluster_8cl_X10_50K_90K_573596_XID.csv" : 8
}


#get measurement data using simulated datasets
for key, val in datasets.items():
    sim_data = pd.read_csv(key, sep = ",")
    last_column = len(sim_data.columns) - 1
    cur_result    = testlm(sim_data.iloc[:,0:last_column], sim_data.iloc[:,last_column], clusters=val)
    cur_resultDF  = pd.DataFrame(cur_result)
    cur_resultDF.to_csv("./testdat/mdat_Kmeans_1h_"+key[key.rindex("/")+1:], sep = ",", header = True, index = False)
"""
cluster_number = 20
sim_data = pd.read_csv("./data_simulation/Cluster_20cl_X4_5K_9K_140246_XID.csv", sep = ",")
last_column = len(sim_data.columns) -1
cur_result    = testlm(sim_data.iloc[:,0:last_column], sim_data.iloc[:,last_column], clusters=cluster_number)
cur_resultDF  = pd.DataFrame(cur_result)
cur_resultDF.to_csv("./testdat/mdat_Kmeans_10min"++".csv", sep = ",", header = True, index = False)
"""