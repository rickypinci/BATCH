# BATCH
BATCH: Adaptive Batching for Efficient MachineLearning Serving on Serverless Platforms
**Deployment:**
To deploy the Lambda serverless function follow the instruction in the following link. However, this code and model provided in the demo does not support batching.

https://aws.amazon.com/blogs/machine-learning/how-to-deploy-deep-learning-models-with-aws-lambda-and-tensorflow/

Our modified models with batching support alog with our modified lambda package are located in the packge director. Our modified lambda packge can be deployed using the similar steps mentioned in the demo just by replacing the model file and packge file. We use Tensorflow 1.8 in our packge. At the time of our experiments this was the only Tensorflow version which could be compated to 50 MB limit of Lambda. 
