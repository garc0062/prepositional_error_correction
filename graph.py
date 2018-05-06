import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle
import DataProcessor as dp

parser = argparse.ArgumentParser(description="Predicting Prepositions")
#model
parser.add_argument('--model_type', dest='model_type', help='type of NN', default='simple')
#size
parser.add_argument('--num_hidden', dest='num_hidden', type=int, help='Number of hidden', default=400)
parser.add_argument('--num_layer', dest='num_layer', type=int, help='type of layers', default=1)
#components
parser.add_argument('--dropout', dest='dropout', type=bool, help='If Dropout', default=False)
parser.add_argument('--keep_prob', dest='keep_prob', type=float, help='Dropout rate', default=1.0)
parser.add_argument('--attention', dest='attention', type=bool, help='If Attention', default=False)
#embedding and features
parser.add_argument('--size_window', dest='size_window', type=int, help='Size window to each size', default=2)
parser.add_argument('--skip_num', dest='skip_num', type=int, help='Number of times a word is used', default=1)
parser.add_argument('--num_sampled', dest='num_sampled', type=int, help='Number of negative examples to sample', default=64)
parser.add_argument('--word_emb_size', dest='word_emb_size', type=int, help='Size of words embedding', default=100)
parser.add_argument('--tags', dest='tags', type=bool, help='If tags as feature', default=False)
parser.add_argument('--tags_emb_size', dest='tags_emb_size', type=int, help='Size of tags embedding', default=20)
parser.add_argument('--parse', dest='parse', type=bool, help='If parse as feature', default=False)
parser.add_argument('--index_emb_size', dest='index_emb_size', type=int, help='Size of indexes embedding', default=10)
parser.add_argument('--lower_limit', dest='lower_limit', type=int, help='Lower limit of instances per tokens', default=0)
parser.add_argument('--under', dest='under', type=bool, help='If undersampling', default=False)
parser.add_argument('--over', dest='over', type=bool, help='If overfitting', default=False)
parser.add_argument('--over_pos', dest='over_pos', type=int, help='number of oversampling', default=0)
parser.add_argument('--under_pos', dest='under_pos', type=int, help='number of undersampling', default=0)
parser.add_argument('--type_input', dest='type_input', help='number of undersampling', default='embedding')
parser.add_argument('--type_data', dest='type_data', help='type of Data', default='trim')
parser.add_argument('--batch', dest='batch_size', type=int, help='Size of batch', default=1024)
parser.add_argument('--spelling', dest='spelling', type=bool, help='If correct spell', default=False)
parser.add_argument('--seq_limit', dest='seq_limit', type=int, help='Limit of sequence at each window side', default=5)
parser.add_argument('--original', dest='original', help='If preposition is included', default='exclude')
parser.add_argument('--class_limit', dest='class_limit', type=int, help='If preposition is included', default=10)
parser.add_argument('--not_include_classes', dest='not_include_classes', nargs='*', help='Classes not to include', default=[])
parser.add_argument('--epochs', dest='epochs', type=int, help='Epochs', default=50)
parser.add_argument('--step', dest='step',  type=int, help='number of step', default=-1)
#System
parser.add_argument('--job-dir', dest='job', help='unnecessary argument to make ml-engine works', default='')
parser.add_argument('--work', dest='work_dir', help='type of layers', default='')
parser.add_argument('--location', dest='location', help='To run locally', default='local')
parser.add_argument('--mode', dest='mode', help='mode of model (train/test/prod)', default='train')
parser.add_argument('-f', dest='f', help='Nothing', default='train')

args = parser.parse_args()

dir = ''

def restoreData(fname):
    if args.original == 'exclude':
        dir = 'classes'
    else:
        dir = 'final'
    object = pickle.load(open("../" + dir + "/" + fname + ".pickle", "rb"), encoding='latin1')
    return object

def cross_validation_loss(title):
    training_loss = restoreData(dp.createName("listTrainLoss", args))
    validation_loss = restoreData(dp.createName("listValLoss", args))
    print(len(training_loss), len(validation_loss))
    max_loss = 1.2 * np.max([training_loss, validation_loss])
    epochs = 10 * ((len(training_loss) - 1)* args.batch_size/int(restoreData(dp.createName("trainSize", args))))
    plt.subplot(2, 1, 1)
    plt.ylabel('Loss')
    #plt.xlabel('Number of Steps')
    plt.title("Cross Validation (Batch=1024, " + title + ", Epochs=" + "{0:.1f}".format(epochs) + ")")
    plt.axis([0, 10*(len(training_loss) - 1), 0, max_loss])
    x_points = [10*i for i in range(len(training_loss))]
    plt.plot(x_points, training_loss, 'r', label="Training (min=" + "{0:.3f}".format(np.min(training_loss)) + ", step=" +
                                                 "{0:.0f}".format(np.argmin(training_loss)*10) + ")")
    plt.plot(x_points, validation_loss, 'b', label="Validation (min=" + "{0:.3f}".format(np.min(validation_loss))+ ", step=" +
                                                 "{0:.0f}".format(np.argmin(validation_loss)*10) + ")")
    plt.legend()

#def cross_validation_accuracy(title):
    training_accuracy = restoreData(dp.createName("listTrainAcc", args))
    print("training", np.max(training_accuracy))
    validation_accuracy = restoreData(dp.createName("listValAcc", args))
    print("validation", np.max(validation_accuracy))

    max_loss = 1.2 * np.max(training_accuracy)
    epochs = 10 * ((len(training_accuracy) - 1) * args.batch_size / int(restoreData(dp.createName("trainSize", args))))
    plt.subplot(2, 1, 2)
    plt.ylabel('Accuracy - Recall')
    plt.xlabel('Number of Steps')
    #plt.title("Accuracy Rate (Batch=1024, " + title + ", Epochs=" + "{0:.1f}".format(epochs) + ")")
    plt.axis([0, 10*(len(training_accuracy) - 1), 0, max_loss])
    x_points = [10*i for i in range(len(training_accuracy))]
    plt.plot(x_points, training_accuracy, colors[2], label="Accuracy Training (max=" + "{0:.2f}".format(np.max(training_accuracy)) + ")")
    plt.plot(x_points, validation_accuracy, colors[3],
             label="Accuracy Validation (max=" + "{0:.2f}".format(np.max(validation_accuracy)) + ", step=" +
                                                 "{0:.0f}".format(np.argmax(validation_accuracy)*10) + ")")
    plt.legend()
    plt.show()

def metrics_validation(title):
    total_steps = len(restoreData(dp.createName("listValLoss", args)))
    recall_all = []
    precision_all = []
    f1_all = []
    f05_all = []
    accuracy_all = []
    for num_step in range(0, total_steps):
        precision_all.append(restoreData(dp.createName("val_" + str(10*num_step) + "corrprec", args)))
        recall_all.append(restoreData(dp.createName("val_" + str(10*num_step) + "corrrec", args)))
        f1_all.append(restoreData(dp.createName("val_" + str(10*num_step) + "corrf1", args)))
        f05_all.append(restoreData(dp.createName("val_" + str(10*num_step) + "corrf05", args)))
        accuracy_all.append(restoreData(dp.createName("val_" + str(10*num_step) + "corracc", args)))

    print(len(f1_all))
    points = [i*10 for i in range(0, total_steps)]
    plt.subplot(2, 1, 1)
    plt.plot(points, precision_all, colors[0], label="Precision (max=" + "{0:.3f}".format(np.max(precision_all))+ ", step=" +
                                                 "{0:.0f}".format(np.argmax(precision_all)*10) + ")")
    #plt.plot(points, recall_all, colors[1], label="Recall")
    plt.plot(points, f1_all, colors[3], label="F1 (max=" + "{0:.3f}".format(np.max(f1_all))+ ", step=" +
                                                 "{0:.0f}".format(np.argmax(f1_all)*10) + ")")
    plt.plot(points, f05_all, colors[4], label="F05 (max=" + "{0:.3f}".format(np.max(f05_all))+ ", step=" +
                                                 "{0:.0f}".format(np.argmax(f05_all)*10) + ")")
    #plt.plot(points, accuracy, colors[4], label="Accuracy")
    plt.title("Metrics for Validation (" + str(title) + ")")
    plt.ylabel('Metrics')
    max_value = 1.2 * np.max([precision_all, f1_all, f05_all])
    plt.axis([0, points[-1], 0, max_value])
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(points, recall_all, colors[4],
             label="Recall (max=" + "{0:.2f}".format(np.max(recall_all)) + ", step=" +
                                                 "{0:.0f}".format(np.argmax(recall_all)*10) + ")")
    plt.plot(points, accuracy_all, colors[5],
             label="Accuracy (max=" + "{0:.2f}".format(np.max(accuracy_all)) + ", step=" +
                   "{0:.0f}".format(np.argmax(accuracy_all) * 10) + ")")
    plt.ylabel('Metrics')
    plt.xlabel('Number of steps')
    max_value = 1.2 * np.max([recall_all, accuracy_all])
    plt.axis([0, points[-1], 0, max_value])
    plt.legend()
    plt.show()

name_preps = {0: 'OF', 1: 'IN', 2: 'FOR', 3: 'TO', 4: 'AS', 5: 'THAT', 6: 'ON', 7: 'FROM', 8: 'WITH', 9: 'BY', 10: 'AT',
              11: 'IF', 12: 'THAN', 13: 'INTO', 14: 'ABOUT', 15: 'BECAUSE', 16: 'LIKE', 17: 'SINCE', 18: 'AFTER', 19: 'THROUGH',
              20: 'OVER', 21: 'DURING', 22: 'WITHOUT', 23: 'ALTHOUGH', 24: 'WHILE', 25: 'WHETHER', 26: 'BEFORE',
              27: 'BESIDES', 28: 'UNDER', 29: 'BETWEEN'}
colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

def roc(experiments, labels, class_prep):

    for ind in range(len(experiments)):
        exec(experiments[ind])
        true_pos_rate = [1] + restoreData(dp.createName("testingrec_class_" + str(class_prep), args)) + [0]
        false_pos_rate = [1] + restoreData(dp.createName("testingfpr_class_" + str(class_prep), args)) + [0]
        plt.plot(false_pos_rate, true_pos_rate, colors[ind], label=labels[ind])
    plt.title("ROC for Class " + str(name_preps[class_prep]))
    plt.ylabel('True Positives Rate')
    plt.xlabel('False Positives Rate')
    plt.axis([0, 1, 0, 1])
    plt.legend()
    #plt.show()

def precision_recall(experiments, labels, class_prep):
    for ind in range(len(experiments)):
        exec(experiments[ind])
        print(dir)
        recall = restoreData(dp.createName("testingrec_class_" + str(class_prep), args))
        precision = restoreData(dp.createName("testingprec_class_" + str(class_prep), args))
        plt.plot(recall, precision, colors[ind], label=labels[ind])
    precision[-1] = precision[-2]
    print(recall)
    print(precision)
    plt.title("Precision - Recall for Class " + str(name_preps[class_prep]))
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.axis([0, 1, 0, 1])
    plt.legend()
    #plt.show()

def metrics_with_threshold(title):
    recall = restoreData(dp.createName("test_a_threc", args))
    precision = restoreData(dp.createName("test_a_thprec", args))
    f1 = restoreData(dp.createName("test_a_thf1", args))
    f05 = restoreData(dp.createName("test_a_thf05", args))
    accuracy = restoreData(dp.createName("test_a_thaccu", args))
    points = [i*(1/100) for i in range(0, 105, 5)]
    plt.subplot(1, 2, 1)
    plt.ylabel('Metrics')
    plt.xlabel('Threshold')
    plt.title("Metrics with threshold (" + str(title) + ")")
    plt.axis([0, 1, 0, 1.2*np.max([precision, f05, f1])])
    plt.plot(points, precision, colors[0], label="Precision (max=" + "{0:.1f}".format(np.max(precision)*100) + "%, threshold=" +
                                                 "{0:.0f}".format(np.argmax(precision)*5) + "%)")
    plt.plot(points, f1, colors[2], label="F1 (max=" + "{0:.1f}".format(np.max(f1)*100) + "%, threshold=" +
                                                 "{0:.0f}".format(np.argmax(f1)*5) + "%)")
    plt.plot(points, f05, colors[3], label="F05 (max=" + "{0:.1f}".format(np.max(f05)*100) + "%, threshold=" +
                                                 "{0:.0f}".format(np.argmax(f05)*5) + "%)")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(points, recall, colors[1], label="Recall (max=" + "{0:.1f}".format(np.max(recall)*100) + "%, threshold=" +
                                                 "{0:.0f}".format(np.argmax(recall)*5) + "%)")
    plt.plot(points, accuracy, colors[4], label="Accuracy (max=" + "{0:.1f}".format(np.max(accuracy)*100) + ", threshold=" +
                                                 "{0:.0f}".format(np.argmax(accuracy)*5) + "%)")
    plt.title("Metrics with threshold (" + str(title) + ")")
    plt.ylabel('Metrics')
    plt.xlabel('Threshold')
    plt.axis([0, 1, 0, 1])
    plt.legend()
    plt.show()

'''
#Number of cells
#1. Training and Validation Loss and Accuracy
#args.keep_prob = ''
experiment = [50, 100, 200, 400, 800, 1600]
dir = 'lists_from_ml'
args.original = 'False'
args.tags_emb_size = ''
args.index_emb_size = ''

for ex in experiment:
    args.num_hidden = ex
    #cross_validation_loss("Hidden=" + str(ex))
    #cross_validation_accuracy("Hidden=" + str(ex))
    metrics_validation("Hidden=" + str(ex))

experiments = ['args.num_hidden = 50', 'args.num_hidden = 100', 'args.num_hidden = 200', 'args.num_hidden = 400',
               'args.num_hidden = 800', 'args.num_hidden = 1600']

#2. Testing
#   a. ROC
#   b. Precision - Recall
labels = ['50', '100', '200', '400', '800', '1600']

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
'''
'''
#4.Precision, Recall, F1, F05, Accuracy with Threshold
#args.mode = 'test'
for ex in experiment:
    args.num_hidden = ex
    metrics_with_threshold("Hidden=" + str(ex))

#Number of layers
#1. Training and Validation Loss and Accuracy
experiment = [2, 4, 8]
dir = 'lists_from_ml'
args.original = 'False'
args.tags_emb_size = ''
args.index_emb_size = ''
for ex in experiment:
    args.num_layer = ex
    cross_validation_loss("Layers=" + str(ex))
    #cross_validation_accuracy("Layers=" + str(ex))
    metrics_validation("Layers=" + str(ex))
    
experiments = ['args.num_layer = 1', 'args.num_layer = 2', 'args.num_layer = 4', 'args.num_layer = 8']
#   a. ROC
#   b. Precision - Recall
args.mode = 'test'
labels = ['1', '2', '4', '8']
for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
    #if (class_id + 1)%4 == 0:
plt.show()

for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
    #if (class_id + 1)%4 == 0:

plt.show()

for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
    #if (class_id + 1)%4 == 0:

plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
#for ex in experiment:
#    args.num_layer = ex
#    metrics_with_threshold("Layer=" + str(ex))
'''
'''
#Dropout
experiment = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
args.num_hidden = 1600
args.num_layer = 8
args.dropout = True

for ex in experiment:
    args.keep_prob = ex
    cross_validation_loss("Dropout=" + str(ex))
    #cross_validation_accuracy("Hidden=" + str(ex))
    metrics_validation("Dropout=" + str(ex))

experiments = ['args.keep_prob = 0.5', 'args.keep_prob = 0.6', 'args.keep_prob = 0.7', 'args.keep_prob = 0.8',
               'args.keep_prob = 0.9', 'args.keep_prob = 1.0']

#2. Testing
#   a. ROC
#   b. Precision - Recall
labels = ['0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
args.mode = 'test'
for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
#args.mode = 'test'
for ex in experiment:
    args.keep_prob = ex
    metrics_with_threshold("Dropout=" + str(ex))
'''

'''
#Tags
experiment = [10, 15, 20, 25]
args.tags = True

for ex in experiment:
    args.tags_emb_size = ex
    cross_validation_loss("Embedding Size=" + str(ex))
    #cross_validation_accuracy("Hidden=" + str(ex))
    metrics_validation("Embedding Size=" + str(ex))

experiments = ['args.tags_emb_size = 10', 'args.tags_emb_size = 15', 'args.tags_emb_size = 20',
               'args.tags_emb_size = 25']

#2. Testing
#   a. ROC
#   b. Precision - Recall
labels = ['10', '15', '20', '25']
args.mode = 'test'
for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
for ex in experiment:
    args.tags_emb_size = ex
    metrics_with_threshold("Embedding Size=" + str(ex))

'''
'''
#Parse
experiment = [10, 15, 20, 25]
args.parse = True

for ex in experiment:
    args.index_emb_size = ex
    cross_validation_loss("Embedding Size=" + str(ex))
    #cross_validation_accuracy("Hidden=" + str(ex))
    metrics_validation("Embedding Size=" + str(ex))

experiments = ['args.index_emb_size = 10', 'args.index_emb_size = 15', 'args.index_emb_size = 20',
               'args.index_emb_size = 25']

#2. Testing
#   a. ROC
#   b. Precision - Recall
labels = ['10', '15', '20', '25']
args.mode = 'test'

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
for ex in experiment:
    args.index_emb_size = ex
    metrics_with_threshold("Embedding Size=" + str(ex))
'''

'''
#Under
dir = 'under'
experiment = [1, 2, 6, 9]
args.under = True

for ex in experiment:
    args.under_pos = ex
    if ex == 6:
        args.original = 'exclude'
    else:
        args.original = 'False'
    cross_validation_loss("Under=" + str(ex + 1))
    #cross_validation_accuracy("Hidden=" + str(ex))
    metrics_validation("Under=" + str(ex + 1))

experiments = ['args.under_pos = 1', 'args.under_pos = 2', 'args.under_pos = 6',
               'args.under_pos = 9']

#2. Testing
#   a. ROC
#   b. Precision - Recall
labels = ['2', '3', '7', '10']

args.mode = 'test'

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
for ex in experiment:
    if ex == 6:
        args.original = 'exclude'
    else:
        args.original = 'False'
    args.under_pos = ex
    metrics_with_threshold("Under=" + str(ex + 1))
'''

'''
#Over
experiment = [0, 1, 2, 6]
args.over = True

for ex in experiment:
    args.over_pos = ex
    cross_validation_loss("Over=" + str(ex + 1))
    #cross_validation_accuracy("Hidden=" + str(ex))
    metrics_validation("Over=" + str(ex + 1))

experiments = ['args.over_pos = 0', 'args.over_pos = 1', 'args.over_pos = 2',
               'args.over_pos = 6']

#2. Testing
#   a. ROC
#   b. Precision - Recall
labels = ['1', '2', '3', '7']
args.mode = 'test'

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

args.mode = 'test'
#4.Precision, Recall, F1, F05, Accuracy with Threshold
for ex in experiment:
    args.over_pos = ex
    metrics_with_threshold("Over=" + str(ex+1))
'''

'''
#Sequence
experiment = [10, 15, 20]
args.original = 'exclude'
dir = 'sequence'

for ex in experiment:
    args.seq_limit = ex
    cross_validation_loss("Sequence=" + str(ex))
    #cross_validation_accuracy("Hidden=" + str(ex))
    metrics_validation("Sequence=" + str(ex))

experiments = ['args.seq_limit = 10', 'args.seq_limit = 15', 'args.seq_limit = 20']

#2. Testing
#   a. ROC
#   b. Precision - Recall
labels = ['10', '15', '20']
args.mode = 'test'

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
for ex in experiment:
    args.seq_limit = ex
    metrics_with_threshold("Sequence=" + str(ex))
'''

'''
#Classes
experiment = [5, 15, 20, 25, 30]
dir = 'testing'

for ex in experiment:
    args.class_limit = ex
    cross_validation_loss("Classes=" + str(ex))
    #cross_validation_accuracy("Hidden=" + str(ex))
    metrics_validation("Classes=" + str(ex))

experiments = ['args.class_limit = 5', 'args.class_limit = 15', 'args.class_limit = 20', 'args.class_limit = 25', 'args.class_limit = 30']

labels = ['5', '15', '20', '25', '30']
args.mode = 'test'

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    if class_id > 4:
        experiments = ['args.class_limit = 15', 'args.class_limit = 20', 'args.class_limit = 25', 'args.class_limit = 30']
        labels = ['15', '20', '25', '30']
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, 12):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(12, 16):
    plt.subplot(2, 2, class_id - 12 + 1)
    if class_id > 14:
        experiments = ['args.class_limit = 20', 'args.class_limit = 25', 'args.class_limit = 30']
        labels = ['20', '25', '30']
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(16, 20):
    plt.subplot(2, 2, class_id - 16 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
experiments = ['args.class_limit = 25', 'args.class_limit = 30']
labels = ['25', '30']
for class_id in range(20, 24):
    plt.subplot(2, 2, class_id - 20 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(24, 28):
    plt.subplot(2, 2, class_id - 24 + 1)
    if class_id > 24:
        experiments = ['args.class_limit = 30']
        labels = ['30']
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(28, 30):
    plt.subplot(2, 2, class_id - 28 + 1)
    if class_id > 24:
        experiments = ['args.class_limit = 30']
        labels = ['30']
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

experiments = ['args.class_limit = 5', 'args.class_limit = 15', 'args.class_limit = 20', 'args.class_limit = 25', 'args.class_limit = 30']

labels = ['5', '15', '20', '25', '30']
args.mode = 'test'

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    if class_id > 4:
        experiments = ['args.class_limit = 15', 'args.class_limit = 20', 'args.class_limit = 25', 'args.class_limit = 30']
        labels = ['15', '20', '25', '30']
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, 12):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(12, 16):
    plt.subplot(2, 2, class_id - 12 + 1)
    if class_id > 14:
        experiments = ['args.class_limit = 20', 'args.class_limit = 25', 'args.class_limit = 30']
        labels = ['20', '25', '30']
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(16, 20):
    plt.subplot(2, 2, class_id - 16 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
experiments = ['args.class_limit = 25', 'args.class_limit = 30']
labels = ['25', '30']
for class_id in range(20, 24):
    plt.subplot(2, 2, class_id - 20 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(24, 28):
    plt.subplot(2, 2, class_id - 24 + 1)
    if class_id > 24:
        experiments = ['args.class_limit = 30']
        labels = ['30']
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(28, 30):
    plt.subplot(2, 2, class_id - 28 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
for ex in experiment:
    args.class_limit = ex
    metrics_with_threshold("Classes=" + str(ex))
'''

'''
#Embeddings
experiment = [50, 200]

dir = 'testing'

for ex in experiment:
    args.word_emb_size = ex
    cross_validation_loss("Embedding=" + str(ex))
    metrics_validation("Embedding=" + str(ex))

experiments = [
"""
args.word_emb_size = 50
args.original = 'exclude'
args.tags_emb_size = '20'
args.index_emb_size = '10'
""",
"""
args.word_emb_size = 100
args.original = 'False'
args.tags_emb_size = ''
args.index_emb_size = ''
""",
"""
args.word_emb_size = 200
args.original = 'exclude'
args.tags_emb_size = '20'
args.index_emb_size = '10'
"""
]

labels = ['50', '100', '200']

args.mode = 'test'

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
for ex in experiment:
    args.word_emb_size = ex
    metrics_with_threshold("Embedding=" + str(ex))
'''

'''
#Spelling
experiment = [True, False]
#dir = 'lists_from_ml'

for ex in experiment:
    args.spelling = ex
    cross_validation_loss("Spelling=" + str(ex))
    metrics_validation("Spelling=" + str(ex))

experiments = [
"""
args.spelling = True
dir = 'testing'
args.tags_emb_size = 20
args.index_emb_size = 10
args.original = 'exclude'
""", """
dir = 'lists_from_ml'
args.spelling = False
args.original = 'False'
args.tags_emb_size = ''
args.index_emb_size = ''
"""]

labels = ['True', 'False']

args.mode = 'test'

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
args.spelling = True
dir = 'testing'
args.tags_emb_size = 20
args.index_emb_size = 10
args.original = 'exclude'
metrics_with_threshold("Spelling=True")

dir = 'lists_from_ml'
args.spelling = False
args.original = 'False'
args.tags_emb_size = ''
args.index_emb_size = ''
metrics_with_threshold("Spelling=False")
'''

'''
#Size window
experiment = [1, 3, 4]
dir = 'testing'

for ex in experiment:
    args.size_window = ex
    cross_validation_loss("Window Size=" + str(ex))
    metrics_validation("Window Size=" + str(ex))

experiments = ['args.size_window = 1', 'args.size_window = 3', 'args.size_window = 4']

labels = ['1', '3', '4']

args.mode = 'test'

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
for ex in experiment:
    args.size_window = ex
    metrics_with_threshold("Window Size=" + str(ex)) 
'''

'''
#Attention
experiment = [True]
dir = 'testing'
args.seq_limit = 20

for ex in experiment:
    args.attention = ex
    cross_validation_loss("Attention=" + str(ex))
    metrics_validation("Attention=" + str(ex))

experiments = [
"""
args.attention = True
args.mode='train'
""","""
args.attention = False
args.mode='test'
"""]

labels = ['True', 'False']

#args.mode = 'test'

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#args.mode='test'
args.original = 'exclude'
#4.Precision, Recall, F1, F05, Accuracy with Threshold
for ex in experiment:
    args.attention = ex
    metrics_with_threshold("Attention=" + str(ex))
'''

'''
#Lower Limit
experiment = [1, 2, 4, 8, 16]
dir = 'lower'

for ex in experiment:
    args.lower_limit = ex
    cross_validation_loss("Lower Limit=" + str(ex))
    metrics_validation("Lower Limit=" + str(ex))

experiments = ['args.lower_limit = 1', 'args.lower_limit = 2', 'args.lower_limit = 4',
               'args.lower_limit = 8', 'args.lower_limit = 16']

labels = ['1', '2', '4', '8', '16']

args.mode = 'test'

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
for ex in experiment:
    args.lower_limit = ex
    metrics_with_threshold("Lower Limit=" + str(ex))
'''

'''
#Bidirectional
experiment = ['bi']
dir = 'bi'

args.mode='train'
for ex in experiment:
    args.model_type = ex
    cross_validation_loss("Model=bidirectional")
    metrics_validation("Model=bidirectional")
        
args.mode='test'
experiments = [
"""
args.model_type = 'bi'
args.original = 'exclude'
args.tags_emb_size = '20'
args.index_emb_size = '10'
""","""
args.model_type = 'simple'
args.original = 'False'
args.tags_emb_size = ''
args.index_emb_size = ''
"""]

labels = ['Bidirectional', 'Simple']

#args.mode = 'test'

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
args.mode = 'test'
args.model_type = 'bi'
args.original = 'exclude'
args.tags_emb_size = '20'
args.index_emb_size = '10'
metrics_with_threshold("Model=bidirectional")
'''
'''
#One-Hot
experiment = ['one_hot']
dir = 'one_hot'

args.mode='train'
for ex in experiment:
    args.type_input = ex
    cross_validation_loss("Input=one-hot")
    metrics_validation("Input=one-hot")

experiments = [
"""
args.type_input = 'one_hot'
args.original = 'exclude'
args.tags_emb_size = '20'
args.index_emb_size = '10'
args.mode = 'train'
""","""
args.type_input = 'embedding'
args.original = 'False'
args.tags_emb_size = ''
args.index_emb_size = ''
args.mode = 'test'
"""]

labels = ['One-Hot', 'Embedding']


for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
args.type_input = 'one_hot'
args.original = 'exclude'
args.tags_emb_size = '20'
args.index_emb_size = '10'
args.mode = 'train'
metrics_with_threshold("Input=one-hot")
'''

'''
#Original
experiment = ['include', 'replace']
dir = 'original'
args.mode='train'
for ex in experiment:
    args.original = ex
    cross_validation_loss("Original=" + str(ex))
    metrics_validation("Original=" + str(ex))

experiments = [
"""
args.original = 'False'
args.tags_emb_size = ''
args.index_emb_size = ''
""","""
args.original = 'replace'
args.tags_emb_size = '20'
args.index_emb_size = '10'
""",
"""
args.original = 'include'
args.tags_emb_size = '20'
args.index_emb_size = '10'
"""
]

labels = ['exclude', 'replace', 'include']
for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, args.class_limit):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold
args.tags_emb_size = '20'
args.index_emb_size = '10'
for ex in experiment:
    args.original = ex
    metrics_with_threshold("Original=" + ex)
'''

#Final
args.lower_limit = 2
args.size_window = 1
args.attention = True
args.spelling = True
args.num_hidden = 1600
args.num_layer = 2
args.tags = True
args.tags_emb_size = 15
args.parse = True
args.index_emb_size = 10
args.original = 'replace'
args.class_limit = 20

dir = 'final'
args.mode='train'
cross_validation_loss("Final")
metrics_validation("Final")

experiments = [
"""
args.original = 'exclude'
args.tags_emb_size = 20
args.index_emb_size = 10
args.lower_limit = 0
args.size_window = 2
args.attention = False
args.spelling = False
args.num_hidden = 400
args.num_layer = 1
args.tags = False
args.parse = False
args.class_limit = 20
""","""
args.lower_limit = 2
args.size_window = 1
args.attention = True
args.spelling = True
args.num_hidden = 1600
args.num_layer = 2
args.tags = True
args.tags_emb_size = 15
args.parse = True
args.index_emb_size = 10
args.original = 'replace'
args.class_limit = 20
"""
]

#experiments = ['']
labels = ['standard', 'final']
for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, 12):
    plt.subplot(2, 2, class_id - 8 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(12, 16):
    plt.subplot(2, 2, class_id - 12 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(16, 20):
    plt.subplot(2, 2, class_id - 16 + 1)
    #roc(experiments, labels, class_id)
    precision_recall(experiments, labels, class_id)
plt.show()

for class_id in range(0, 4):
    plt.subplot(2, 2, class_id + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(4, 8):
    plt.subplot(2, 2, class_id - 4 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(8, 12):
    plt.subplot(2, 2, class_id - 8 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(12, 16):
    plt.subplot(2, 2, class_id - 12 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()
for class_id in range(16, 20):
    plt.subplot(2, 2, class_id - 16 + 1)
    roc(experiments, labels, class_id)
    #precision_recall(experiments, labels, class_id)
plt.show()

#4.Precision, Recall, F1, F05, Accuracy with Threshold

metrics_with_threshold("Final")