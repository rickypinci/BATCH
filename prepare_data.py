import sys
import numpy as np
import pandas as pd
import glob
import math



def new_reading(name):
    latency = []
    batch = []
    with open(name,"r") as lines:
        for line in lines:
            final = line.split()
            latency.append(float(final[3]))
	    batch.append(float(final[1]))
    print("std:{}".format(np.std(latency)/np.mean(latency)))
    print("95 percentile:{}".format(np.percentile(latency, 95)))
    return (np.mean(batch),np.mean(latency))

def plot_one_cdf(latency,file_name):
   num_bins = 20 
   counts, bin_edges = np.histogram(latency, bins=num_bins, normed=True)
   cdf = np.cumsum(counts)
   with open ("cdf_{}".format(file_name),"w") as fp:
       for i in range (len(cdf)):
           fp.write("{}\t{}\n".format(bin_edges[i+1],cdf[i]/cdf[-1]))

def read_latecy_per_request(file_name):
    mylatecny = []
    with open(file_name,"r") as lines:
        for line in lines:
            parts = line.split()
            my_val = float(parts[1])
            mylatecny.append(my_val)
    plot_one_cdf(mylatecny, file_name )
    return np.mean(mylatecny)

def cost_calculation(memory_util, billing_time):
    cost = []
    init_prices = 208
    for i in range (len(memory_util)):
	billing = billing_time[i]
        temp = memory_util[i] - 128 
        temp = math.ceil(temp/64)
        total_compute_gb = temp*104 + 208
        cost.append(billing * (total_compute_gb/1024))
    return cost
def cost_per_request(latency_per_batch, measured_batch_size):
    memory_cost = 4.897  # cost per million request with 100 msec duration with 3008MB memory allocated
    GB_s = 2.9375 # 3008/1024
    all_cost = []
    for i in range(len(latency_per_request)):
        billing1 = latency_per_request[i]//100
        billing = (billing1 +1)/10
        cost = memory_cost * GB_s * billing
        print("billing {} cost {}".format(billing, cost))
        all_cost.append(cost/(2*float(measured_batch_size[i])))
    #print(all_cost)  
    return all_cost 
def write_results(inter_arrival, results, file_name):
    with open(file_name,"w") as f:	
        for i in range (len(inter_arrival)):
	 #print(val)
	     f.write("{} \t {}".format(str(inter_arrival[i]),str(results[i])))
	     f.write('\n')
    f.close()

def read_measured_bs(file_name):
    bs = []
    with open(file_name,"r") as lines:
        for line in lines:
            final = line.split()
            bs.append(float(final[0]))
    return (np.mean(bs))

if __name__ =="__main__":

     batch = sys.argv[1]
     time_out = sys.argv[2]
     latencies = []
     latency_per_request = []
     memory_util = []
     all_billing = []
     measured_bs = []
     cost_pre_request = []
     print("read file")
     inter_arrival = [10]
      #[0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3]
      #Latency_per_request_batch_1_inter_arrival_22_time_out1
     for c in inter_arrival:
        #c = float(1/float(c))
        file_name = 'Latency_perbatch_batch_{}_inter_arrival_{}_time_out{}_.log'.format(batch,c, time_out)
        latency_per_request_filename = 'Latency_per_request_batch_{}_inter_arrival_{}_time_out{}_.log'.format(batch,c,time_out)
	print(file_name)
        measured_batch, latency = new_reading(file_name)
        latencies.append(float(latency))#/int(measured_bs[-1])
        latency_per_request.append(read_latecy_per_request(latency_per_request_filename))
        measured_bs.append(measured_batch)
   
     cost_pre_request = cost_per_request(latencies,measured_bs)
     print(cost_pre_request)
     latency_file = "latency_batch_size_{}_timeout_{}".format(batch,time_out)
     actual_bs_file = "measured_batch_size_{}_timeout_{}".format(batch,time_out)
     latency_per_request_file = "latency_per_request_batch_size_{}_timeout_{}".format(batch,time_out)
     cost_per_request_file = "cost_per_request_batch_size_{}_timeout_{}".format(batch,time_out)
     write_results(inter_arrival, latencies,latency_file)
     write_results(inter_arrival, measured_bs, actual_bs_file)
     write_results(inter_arrival, latency_per_request, latency_per_request_file)
     print ("{},{}".format(len(cost_pre_request), len(inter_arrival)))
     write_results(inter_arrival, cost_pre_request, cost_per_request_file)
     #print(math.ceil(billing/100)*100)
     #print(memory)	
