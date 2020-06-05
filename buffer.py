import argparse
import boto3
import threading
import time
from time import sleep
import numpy as np
from datetime import datetime
import sys
import base64
import json
import serverless

class MyRequest():
	
	def __init__(self, arrival_time = 0 , end_time=0, latency=0):
		self.arrival_time = arrival_time
		self.end_time = end_time
		self.latency = end_time
		self.wait_time = 0

	def my_print(self):
		print("start = {} end = {} latency =- {}".format(self.arrival_time, self.end_time, self.latency))

def read_trace_arrival (my_trace):
	my_arrival = []
	with open(my_trace, 'r') as fp:
		for line in fp:
			val = line.strip('\n')
			my_arrival.append(float(val))
	return my_arrival
        
def generate_request(batch_size, time_out, arrival_process, inter_arrival, trace_path, function_name):
	
    my_queue = []
    batch_size = batch_size
    count = 0
    total_delay = 0.0
    timestamp = 0.0
    t_out = []
    bs = []
    trace_arrival = []
    delay = 0.0

    if 'exp' not in arrival_process:
        print("Runnng experiment with traces from {}".format(trace_path))
        trace_arrival =  read_trace_arrival(trace_path)
    else:
        print("Runnng experiment with inter arrival time from {}".format(inter_arrival))
	

	#Change the value in for number of requests
    for i in range (15): 
        my_queue.append(MyRequest((total_delay)*1000,0,0))
        count +=1
        if 'exp' in arrival_process:
            delay = np.random.exponential(scale=(float(1/float(inter_arrival))), size=None) 
        else:   
            delay = trace_arrival [i%2500000] 

        if count < batch_size and total_delay + delay <= float(time_out):
            sleep(delay)
            total_delay += delay
        elif count == batch_size:
            t_out.append(total_delay)
            bs.append(count)
            request_list = []
            while len(my_queue) > 0:
                request = my_queue.pop(0)
                request.wait_time = (total_delay *1000) - request.arrival_time
                request_list.append(request)
            serverless.MyServerless(count, float(total_delay)*1000, request_list, batch_size, inter_arrival, time_out, function_name).start()
            count = 0
            total_delay = 0.0
            sleep(delay)
        else:
            sleep(time_out - total_delay)
            total_delay = float(time_out)
            delayRemainder = delay - (time_out - total_delay)
            t_out.append(total_delay)
            bs.append(count)
            request_list = []
            while len(my_queue) > 0:
                request = my_queue.pop(0)
                request.wait_time = (total_delay *1000) - request.arrival_time
                request_list.append(request)
            serverless.MyServerless(count, float(total_delay)*1000, request_list, batch_size, inter_arrival, time_out, function_name).start()
            count = 0
            total_delay = 0.0
            sleep(delayRemainder)

        timestamp += delay

    print("mean batch size {}".format(np.mean(bs)))

def get_args():
    parser =argparse.ArgumentParser()
    parser.add_argument('--function_name', type = str, default = 'Inception-V4', help = "Name of of deployed lambda function")
    parser.add_argument('--memory', type = str, default = '3008', help = "Integer value between 512 to 3008 to allocate memory for lambda function")
    parser.add_argument('--batch_size', type = int, default = 5, help = "Batch size must be an integer")
    parser.add_argument('--time_out', type = float, default = 1.0, help = "Timeout value must be a float in seconds")
    parser.add_argument('--arrival_process', type = str,default = str, help = "Traces or exponential arrival process")
    parser.add_argument('--inter_arrival', type = float, default = 20.0, help = "Float value to generate request per seconds in case of exponential arrival")
    parser.add_argument('--trace_path', type = str, default = 'arrivals/MMPP_arrival15', help = "Incase of trace provide a trace path")
    return parser.parse_args()

if __name__ == '__main__':
    
    my_args = get_args()
   
    print("Starting Experiment for") 
    print("batch = {}\tarrival process = {}\ttiem out = {}\n".format(my_args.batch_size, my_args.arrival_process, my_args.time_out))
    response = mylambda.update_function_configuration(FunctionName = 'mxnet-lambda-v2', MemorySize=my_args.memory)
    generate_request(my_args.batch_size, my_args.time_out, my_args.arrival_process, my_args.inter_arrival, 
                    my_args.trace_path, my_args.function_name)
    print("Done batch = {}\tarrival process = {}\ttiem out = {}\n".format(my_args.batch_size, my_args.arrival_process, my_args.time_out))
    

    '''batch_size = [1] #,5,10,15,20] 
    time_out = [1.0] # in seconds 
    mylambda = boto3.client('lambda')
    inter_arrivals = [20] # request per seconds
    function_name = 'mxnet-lambda-v2'
    trace_path = 'traces/MMPP_arrival'
    arrival_process= 'trace'
    
    for inter_arrival in inter_arrivals:
        for batch in batch_size:
            for tout in time_out:
                print("batch = {}\tinter_arrival = {}\ttiem out = {}\n".format(batch, inter_arrival, tout))
                generate_request(batch, tout, arrival_process, inter_arrival, trace_path, function_name)
                print(" Done batch = {}\tinter_arrival = {}\ttiem out = {}\n".format(batch, inter_arrival, tout))
                sleep(30)
    '''