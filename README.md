# BATCH
BATCH: Adaptive Batching for Efficient MachineLearning Serving on Serverless Platforms


**prerequisite**

- [AWS  Cli](https://aws.amazon.com/cli/)
- [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

---
**Deployment:**

To deploy the Lambda serverless function follow the instruction in the [How to deploy deep learning models with aws lambda and tensorflow](https://aws.amazon.com/blogs/machine-learning/how-to-deploy-deep-learning-models-with-aws-lambda-and-tensorflow/). However, this tutorial do not support batching.


Our updated [model and lambda package](https://drive.google.com/drive/folders/1R5eJ-dQZDmTU45-YBj1CJyiYWsExTWvN?usp=sharing) supports batching. The batching enabled lambda packge can be deployed by following the same steps mentioned in the demo and just by replacing the model file and packge file. We use Tensorflow 1.8 in our packge. At the time of our experiments this was the most latest Tensorflow version which could be zipped to 50 MB (Lambda limitation). 



-----
**Find optimal configuration**
- To find the optimal configuration of the serverless environment, run the _solver.py_ python script.
- _solver.py_ must be run with python3 and requires the following modules:
   1. argparse
   2. numpy
   3. matplotlib
   4. scipy
   5. ortools
- To print the help, try: _python solver.py --help_
- To run the solver, try: _python solver.py --model TF-inceptionV4 --percentile 0.95 --slo 0.00003 --constraint cost --trace Twitter --start 1 --end 1_
---
**Run Experiments**


Usage:
python buffer.py.py (default setting)

- Run experiment with Exponential arrival python buffer.py.py (default setting).
   1. In default setting it will run experiments  for expontential arrival.
   2. Memory size is set to 3008 MB.
   3. Workload intensity is set to 20 request per second.
   4. Batch size value in serveless.py (default 5).
   5. Timeout values can also be varied (default 1 second).
   
-Run with exponential arrival:

python buffer.py --batch_size 5 --time_out 1 --arrival_process exp --inter_arrival 10 --function_name inception-v4 --memory 3008 
- Run experiments using a trace python serverless.py trace $tracelocation
   1. These experiments are conducted to evaluate the performance of the model in terms of latency as well as cost.
   python buffer.py --batch_size 5 --time_out 1 --arrival_process trace --trace_path ./traces/MMPP_arrival --function_name inception-v4 --memory 3008 

----
**Run model**
- Model takes the arrival process i.e. inter-arrival time to calculate the efficient memory size, batch size and batch timeout. 
-----
**Collect logs**
- Once the experiments are done three log files are generated for each experiment.
  1. Lambda logs: These logs contains all the information regarding each lambda invocation i.e. print out values in the lambda function, init time, execution time, billing time, memory utilization, exceptions if any and error if any.
  2. Lambda per batch logs: These logs contains information regardin batch i.e. Batch starting time, Batch ending time, Batch size and Batch serivce time.
  3. Lmabda pre request logs: This file contains all the information of each request i.e. arrival time, departure time, latency, and size of batch it was served in.
  
  
  **All log are collected through cloudwatch default configurations**
