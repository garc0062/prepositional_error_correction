# prepositional_error_correction

The project is designed to test different recurrent neural network approaches on the 
task of detecting and correcting prepositional errors made by English learners as a second language.

The code is designed to run on the Google ML-Engine, and it uses a set of parameters to run the tests. run --help for more information.

The code needs the NUCLE corpus provided by ConLL 2013 shared task to run. Altough it can work using other corpora, the code must be modified.

Components:

task: contains the main flow.

NN_models: contain methods for running different models.

DataManager: a package providing functions to manage the examples.

DataProcessor: a package providing function to generate examples, in addition to other utilities.

Embedding: contains code to generte embedding layers.

ErrorCorrectionEvaluation: cotains functions to generate metrics to evaluate the algorithm.

graph: a separate utility to create graphs such as precision-recall and ROC.
