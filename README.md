# HomeWork_W1_DL_ICT

## Introduction 
This is a simple implementation of MLP experiment for my homework in deep learning course. The problem is to
classify the generated two dimensional point as 1 if they lie inside a circle.

## Repository Usage
Install the environment as following

```
conda create --name homework1 python==3.7
pip install requirements.txt
```


## Data
The data can be illustrated via the following image 

![Data](/img_vis_data_train.png)

## Experiments

There are five experiments in total:
+  MLP with one hidden layer, 3 perceptrons, L2 loss, SGD optimizer, no dropout, 10 epoches
+ MLP with one hidden layer, 3 perceptrons, L2 loss, SGD optimizer, no dropout, 500 epoches
+  MLP with one hidden layer, 128 perceptrons, CE loss, Adam optimizer, no dropout, 100 epoches
+  MLP with three hidden layer, (32, 64, 32) perceptrons, CE loss, Adam optimizer, no  dropout, 100 epoches
+ MLP with three hidden layer, (32, 64, 32) perceptrons, CE loss, Adam optimizer, 20%  dropout, 100 epoches

Since github is not a good place to display my plotted graphs and my comments, please help me to check [Graph plots and Reports](https://wandb.ai/nttung1110/Official_v5_HomeworkDL/reports/Comment-about-the-experiments--VmlldzoxNDc0Mzgy?accessToken=17p3ohks8e48svl1hh34g0gww83t1a9d6hm7kudhoq0dukvsdl9u77gyttok4se3) to
visualize the graph being structured in nice format.
