import boto3
import threading
import time
from time import sleep
import numpy as np
from datetime import datetime
import sys
import base64
import json
class MyRequest():
	
	def __init__(self, arrival_time = 0 , end_time=0, latency=0):
		self.arrival_time = arrival_time
		self.end_time = end_time
		self.latency = end_time
		self.wait_time = 0

	def my_print(self):
		print("start = {} end = {} latency =- {}".format(self.arrival_time, self.end_time, self.latency))


class MyServerless(threading.Thread):
	
	def __init__ (self, batch_size, queue_time, request_list, actual_batch_size, inter_arrival, time_out):
		threading.Thread.__init__(self)
		self.batch_size = batch_size
		self.queue_time = queue_time
		self.request_list = request_list
		self.actual_batch_size = actual_batch_size
		self.time_out = time_out
		self.inter_arrival = inter_arrival
		self.service_time = {1:1734.26, 2:2313.26, 3:2875.27, 4:3398.42, 5:3934.34,6:4531.25, 7:5047.03, 8:5581.86, 9:6079.07, 10:6724.86, 11:7292.53, 12:7807.46, 13:8375.48, 14:8985.00, 15:9542.51,16:10100.73, 17:10595.58, 18:11476.64, 19:13429.97, 20:14089.95}
		
	
	def send_request(self):
		mylambda =  boto3.client('lambda')
		mutex_lock2 = threading.Lock()
		#data= {"batch":"{}".format(self.batch_size)}
		data= {"BS":self.batch_size}
		batch_service_time = mylambda.invoke(FunctionName='mxnet-lambda-v2', InvocationType = 'RequestResponse', LogType = 'Tail', Payload=json.dumps(data))
		file_lambda_logs = "Lambda_logs_batch_{}_inter_arrival_{}_time_out{}_.log".format(self.actual_batch_size, self.inter_arrival,self.time_out) # "exp"
		mutex_lock2.acquire()
		with open (file_lambda_logs, "a+") as fl:
			fl.write("{}\n".format(base64.b64decode(batch_service_time['LogResult'])))
		fl.close()
		mutex_lock2.release()
		return batch_service_time['Payload'].read()

		

	def run(self):
		
		file_batch_latency = "Latency_perbatch_batch_{}_inter_arrival_{}_time_out{}_.log".format(self.actual_batch_size, self.inter_arrival,self.time_out) # "exp"
		file_request_latency = "Latency_per_request_batch_{}_inter_arrival_{}_time_out{}_.log".format(self.actual_batch_size, self.inter_arrival,self.time_out) # "exp"
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
				fp.write("Request_latency\t{}\tbatch_size\t{}\t\ttimestamp\t{}\n".format(request.latency,self.batch_size, int(my_clock_time + request.wait_time)))
				mutex_2.release()
		fp.close()


def read_mmpp_arrival ():
	my_arrival = []
	with open('arrivals/MMPP_arrival15', 'r') as fp:
		for line in fp:
			val = line.strip('\n')
			my_arrival.append(float(val))
	return my_arrival

def generate_request(batch_size, inter_arrival, time_out):
	my_queue = []
	batch_size = batch_size
	count = 0
	total_delay = 0.0
	timestamp = 0.0
	t_out = []
	bs = []
	#MMPP_arrival = []
	#MMPP_arrival =  read_mmpp_arrival() # Uncomment for MMPP2 arrival process
	flag_count = False
        print(float(1/float(inter_arrival)))
        total_req = inter_arrival*3600
	for i in range (2000): #total_req 1*batch_size #
		my_queue.append(MyRequest((total_delay)*1000,0,0)) #int(time.time())
		count +=1
		delay = np.random.exponential(scale=(float(1/float(inter_arrival))), size=None) # Uncomment for Exponential arrival process
		#delay = MMPP_arrival [i%2500000] # Uncomment for MMPP2 arrival process

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
			MyServerless(count, float(total_delay)*1000, request_list, batch_size, inter_arrival, time_out).start()
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
			MyServerless(count, float(total_delay)*1000, request_list, batch_size, inter_arrival, time_out).start()
			count = 0
			total_delay = 0.0
			sleep(delayRemainder)

		timestamp += delay

	print("mean batch size {}".format(np.mean(bs)))

if __name__ == '__main__':
	print("This is my start time")
	batch_size = [1,5,10,15,20] #, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
	time_out = [1.0]#[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
	mylambda = boto3.client('lambda')
	inter_arrivals = [20]#[10, 15, 20, 25, 30, 35, 40, 45,50, 55, 60, 65, 70, 75, 80, 85, 90, 95 ,100]
        response = mylambda.update_function_configuration(FunctionName = 'mxnet-lambda-v2', MemorySize=3008)
	for inter_arrival in inter_arrivals:
		for batch in batch_size:
			for tout in time_out:
                                print("batch = {}\tinter_arrival = {}\ttiem out = {}\n".format(batch, inter_arrival, tout))
				generate_request(batch, inter_arrival, tout)
                                print(" Done batch = {}\tinter_arrival = {}\ttiem out = {}\n".format(batch, inter_arrival, tout))
                                sleep(30)
