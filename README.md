# BATCH
BATCH: Adaptive Batching for Efficient MachineLearning Serving on Serverless Platforms


**prerequisite**

- AWS  Cli  (https://aws.amazon.com/cli/)
- boto3(https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

---
**Deployment:**

To deploy the Lambda serverless function follow the instruction in the following link. However, this tutorial do not support batching.

https://aws.amazon.com/blogs/machine-learning/how-to-deploy-deep-learning-models-with-aws-lambda-and-tensorflow/

Our updated model and lambda package with batching enabled are located in the link below. Our modified lambda packge can be deployed using the similar steps mentioned in the demo just by replacing the model file and packge file. We use Tensorflow 1.8 in our packge. At the time of our experiments this was the only Tensorflow version which could be zipped to 50 MB (Lambda limitation). 

https://drive.google.com/drive/folders/1g7An2M7bIVJhdUFQESInCX5zokS0EF0r?usp=sharing

-----
**Find optimal configuration**
- To find the optimal configuration of the serverless environment, run the _solver.py_ python script.
- _solver.py_ must be run with python3 and requires the following modules:
   1. argparse
   2. numpy
   3. matplotlib
   4. scipy
   5. ortools
- To print the help, try _python solver.py --help_
- To run the solver, try _python solver.py --model TF-inceptionV4 --percentile 0.95 --slo 0.00003 --constraint cost --trace Twitter --start 1 --end 1_
---
**Run Experiments**
- Run experiment with Exponential arrival python serverless.py exp (default setting).
   1. These experiments are conducted to generate the profiling data to train the model for prediction.
   2. Memory size can be adjusted for each experiments by varying the memory value in Serverless.py (default 3008 MB).
   3. To change the workload intensity set the value to inter_arrival in Serverless.py ( default 20 request per second).
   4. Adjust the batch size value in serveless.py (default 5).
   5. Timeout values can also be varied (default 1 second).
   
- Run experiments using a trace python serverless.py trace $tracelocation
   2. These experiments are conducted to evaluate the performance of the model in terms of latency as well as cost.
----
**Run model**
- Model takes the arrival process i.e. inter-arrival time to calculate the efficient memory size, batch size and batch timeout. 
-----
**Collect logs**
- Once the experiments are done three log files are generated for each experiment.
  1. Lambda logs: These logs contains all the information regarding each request i.e. print out values in the lambda function, init time, execution time, billing time, memory utilization, exceptions if any and error if any.
  2. Lambda per batch logs: These logs contains information regarding logs i.e. Batch starting time, Batch ending time, Batch size and Batch serivce time.
  3. Lmabda pre request logs: This file contains all the information of each request i.e. arrival time, departure time, latency, and size of batch it was served in.
  
  
  **All log are collected through cloudwatch default configurations**
