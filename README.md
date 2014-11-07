spark-ml
========

Example pipelines with new interfaces 

To build this clone https://github.com/mengxr/spark and checkout the branch SPARK-3530
Run sbt/sbt publish-local on the SPARK-3530 branch and then build this project.

To run the MNIST pipeline 
1. Download the test and train data from
   http://s3.amazonaws.com/mnist-data/train-mnist-dense-with-labels.data
   http://s3.amazonaws.com/mnist-data/test-mnist-dense-with-labels.data 

2. Run the pipeline using
  ./sbt/sbt \ 
    -Djava.library.path=/Users/shivaram/debian-shared/spark-ml/lib \
    "run-main ml.MnistRandomFFTPipeline local data/train-mnist-dense-with-labels.data data/test-mnist-dense-with-labels.data 1"
