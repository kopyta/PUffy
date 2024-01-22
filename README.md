# PUffy
*Put a label on it*

PUffy is a Python library designed for creating the best possible models given a dataset with positive-unlabeled data. It implements two-step classification approach. The package is also suitable for considerations on PU classification with fully labeled dataset, as the package presents labeling methods to artificially mimic labeling mechanisms resulting in positive-unlabeled data. Puffy consists of four modules, namely labelling_methods, rn_methods, training and pu_pipeline.

So far, it offers:

- 4 labeling methods:
   one method assuming SCAR scenario,

  - three methods being variants of SAR setting,

- 3 techniques for identifying reliable-negatives sample from the unlabeled data:
  - a technique based on KMeans,
  
  - a technique based on KNN,
  
  - a technique based on Spy analysis,
  
  - an iterative classification model for identifying reliable negatives, using Logistic Regression classification,

- visualizations of the model performance,

- a naive classification method for comparison,

- a pipeline, with which presented experiments were conducted.

As of today, the package is meant for binary classification problems.
