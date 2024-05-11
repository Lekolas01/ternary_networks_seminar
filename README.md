# Practical Work in AI - Ternary Weight Networks

This Work will be composed of the implementation of the following two papers concerning Quantization of Neural Networks into Ternary Weight Networks:
* Sparsity-Control Ternary Weight Networks - https://arxiv.org/abs/2011.00580
* LQ-Nets: Learned Quantization for Highly Accurate and Compact Deep Neural Networks - https://arxiv.org/abs/1807.10029

The goal is to implement them on the basis of the pytorch framework and compare them.

Another goal would be to try and think about how transform Ternary weight networks into some kind of simpler, symbolic model. Ultimately, the goal is to create a model with a good performance on categorical datasets like: 
* the adult dataset - https://archive.ics.uci.edu/ml/datasets/adult, and/or
* the housing dataset - https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Currently the script main.py contains the full pipeline of loading the data, then training the model to saving the trained model parameters.

This rule learning suite contains the following callable python scripts:
* 