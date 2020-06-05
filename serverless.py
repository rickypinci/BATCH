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

class MyServerless(threading.Thread):
	
	def __init__ (self, batch_size, queue_time, request_list, actual_batch_size, inter_arrival, time_out, function_name):
		threading.Thread.__init__(self)
		self.batch_size = batch_size
		self.queue_time = queue_time
		self.request_list = request_list
		self.actual_batch_size = actual_batch_size
		self.time_out = time_out
		self.inter_arrival = inter_arrival
		self.function_name = function_name
		self.service_time = {1:1734.26, 2:2313.26, 3:2875.27, 4:3398.42, 5:3934.34,6:4531.25, 7:5047.03, 8:5581.86, 
		            9:6079.07, 10:6724.86, 11:7292.53, 12:7807.46, 13:8375.48, 14:8985.00, 15:9542.51,16:10100.73, 
					17:10595.58, 18:11476.64, 19:13429.97, 20:14089.95}
		
	
	def send_request(self):
		mylambda =  boto3.client('lambda')
		mutex_lock2 = threading.Lock()
		#data= {"batch":"{}".format(self.batch_size)}
		data= {"BS":self.batch_size}
		batch_service_time = mylambda.invoke(FunctionName=self.function_name, InvocationType = 'RequestResponse', 
		            LogType = 'Tail', Payload=json.dumps(data))
		file_lambda_logs = "Lambda_logs_batch_{}_inter_arrival_{}_time_out{}_.log".format(self.actual_batch_size, 
		            self.inter_arrival,self.time_out) # "exp"
		mutex_lock2.acquire()
		with open (file_lambda_logs, "a+") as fl:
			fl.write("{}\n".format(base64.b64decode(batch_service_time['LogResult'])))
		fl.close()
		mutex_lock2.release()
		return batch_service_time['Payload'].read()

		

	def run(self):
		
		file_batch_latency = "Latency_perbatch_batch_{}_inter_arrival_{}_time_out{}_.log".format(self.actual_batch_size,
				      self.inter_arrival,self.time_out) # "exp"
		file_request_latency = "Latency_per_request_batch_{}_inter_arrival_{}_time_out{}_.log".format(self.actual_batch_size,
			              self.inter_arrival,self.time_out) # "exp"
		mutex_1 = threading.Lock()
		mutex_2 = threading.Lock()
		batch_latency = float(self.send_request())
		
		mutex_1.acquire()
		with open (file_batch_latency, "a+") as fl:
			fl.write("batch\t{}\tlatency\t{}\n".format(self.batch_size, batch_latency))
		fl.close()
		mutex_1.release()

		with open (file_request_latency,"a+") as fp:
                        my_clock_time = int(time.time()*100)
			for request in self.request_list:
				request.latency = request.wait_time + float(batch_latency)
				mutex_2.acquire()
				fp.write("Request_latency\t{}\tbatch_size\t{}\t\ttimestamp\t{}\n".format(request.latency,
					self.batch_size, int(my_clock_time + request.wait_time)))
				mutex_2.release()
		fp.close()