from __future__ import division
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 21:55:34 2018

@author: pablo
"""
import math
import numpy as np
import DataProcessor as dp
from tensorflow.python.lib.io import file_io
import pickle

'''
Given an array of values, it computes its probability distribution.
'''
def softmax(set_values):
    exp_values = np.exp(set_values - np.max(set_values))
    return exp_values / exp_values.sum()

'''
WAS Evaluation Schema by Chodorow et al. [2012]
'''
def wasEvaluationSchema(examples, output, prefix, args):
    output.write("WAS Evaluation Schema\n\t")
    #Detection
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total = len(examples)

    #DETECTION
    for i in range(total):
        student_preposition = np.argmax(examples[i].original)
        correct_preposition = np.argmax(examples[i].correct)
        predictor_preposition = np.argmax(examples[i].predicted)

        '''if student_preposition >= args.class_limit:
            if student_preposition == correct_preposition:
                true_negative += 1
            else:
                false_negative += 1
            continue'''

        #if predictor_preposition == 9:
        #    output.write("Another Prep \n")
        #    predictor_preposition = student_preposition

        # Error is not flagged, and error does not exist
        if student_preposition == predictor_preposition and correct_preposition == predictor_preposition:
            true_negative = true_negative + 1
        # Flags a possible error, but there is not error or correction is wrong
        elif student_preposition != predictor_preposition and correct_preposition == student_preposition:
            false_positive = false_positive + 1
        # Does not flag error, but error exists
        elif student_preposition == predictor_preposition and correct_preposition != predictor_preposition:
            false_negative = false_negative + 1
        # Flags a possible error, and correction is right
        elif student_preposition != predictor_preposition and correct_preposition == predictor_preposition:
            true_positive = true_positive + 1
        #flag an error but currection is wrong
        elif student_preposition != predictor_preposition and correct_preposition != predictor_preposition:
            false_negative += 1
            false_positive += 1
    metrics(true_positive, false_positive, true_negative, false_negative, output, prefix + "corr", args)

'''
WAS Evaluation Schema by Chodorow et al. [2012]
With threshold
'''
def wasEvaluationSchemaWithThreshold(examples, output, prefix, args):
    output.write("WAS Evaluation Schema With Threshold\n\t")
    #Detection
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total = len(examples)
    #Correction
    precision = []
    recall = []
    f1 = []
    f05 = []
    accuracy = []

    for threshold in range(0, 105, 5):
        for i in range(total):
            #Get indexes
            student_preposition = np.argmax(examples[i].original)
            correct_preposition = np.argmax(examples[i].correct)
            predictor_preposition = np.argmax(examples[i].predicted)
            #Only if the confidence of the predictor is as big as 90%, the prediction is taken into account.
            predictions = softmax(examples[i].predicted)

            max_prediction = np.max(predictions)
            if(100 * max_prediction < threshold):
                predictor_preposition = student_preposition
            #If outside classes
            '''if student_preposition >= args.class_limit:
                if student_preposition == correct_preposition:
                    true_negative += 1
                else:
                    false_negative += 1
                continue'''

            # Error is not flagged, and error does not exist
            if student_preposition == predictor_preposition and correct_preposition == predictor_preposition:
                true_negative = true_negative + 1
            # Flags a possible error, but there is not error or correction is wrong
            elif student_preposition != predictor_preposition and correct_preposition == student_preposition:
                false_positive = false_positive + 1
            # Does not flag error, but error exists
            elif student_preposition == predictor_preposition and correct_preposition != predictor_preposition:
                false_negative = false_negative + 1
            # Flags a possible error, and correction is right
            elif student_preposition != predictor_preposition and correct_preposition == predictor_preposition:
                true_positive = true_positive + 1
            #flag an error but currection is wrong
            elif student_preposition != predictor_preposition and correct_preposition != predictor_preposition:
                false_negative += 1
                false_positive += 1

        precision.append(getPrecision(true_positive, false_positive))
        recall.append(getRecall(true_positive, false_negative))
        f1.append(getF1(recall[len(precision) - 1], precision[len(precision) - 1]))
        f05.append(getF05(recall[len(precision) - 1], precision[len(precision) - 1]))
        accuracy.append(getAccuracy(true_positive, false_positive, true_negative, false_negative))

    saveMetric(args.work_dir + "lists/" + prefix + "prec", precision, args)
    saveMetric(args.work_dir + "lists/" + prefix + "rec", recall, args)
    saveMetric(args.work_dir + "lists/" + prefix + "f1", f1, args)
    saveMetric(args.work_dir + "lists/" + prefix + "f05", f05, args)
    saveMetric(args.work_dir + "lists/" + prefix + "accu", accuracy, args)

'''
Evaluates error detection.
'''
def evaluateErrorDetection(examples, output, prefix, args):
    output.write("Detection... \n\t")
    #Detection
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total = len(examples)

    for i in range(total):
	    #Get indexes
        student_preposition = np.argmax(examples[i].original)
        correct_preposition = np.argmax(examples[i].correct)
        predictor_preposition = np.argmax(examples[i].predicted)

        '''if student_preposition >= args.class_limit:
            if student_preposition == correct_preposition:
                true_negative += 1
            else:
                false_negative += 1
            continue'''

        # Error is not flagged, and error does not exist
        if student_preposition == predictor_preposition and correct_preposition == predictor_preposition:
            true_negative = true_negative + 1
        # Flags a possible error, but there is not error or correction is wrong
        elif student_preposition != predictor_preposition and correct_preposition == student_preposition:
            false_positive = false_positive + 1
        # Does not flag error, but error exists
        elif student_preposition == predictor_preposition and correct_preposition != predictor_preposition:
            false_negative = false_negative + 1
        # Flags a possible error, and correction is right
        elif student_preposition != predictor_preposition and correct_preposition == predictor_preposition:
            true_positive = true_positive + 1
        elif student_preposition != predictor_preposition and correct_preposition != predictor_preposition:
            true_positive += 1
    metrics(true_positive, false_positive, true_negative, false_negative, output, prefix + "det", args)
    
'''
WAS Evaluation Schema by Chodorow et al. [2012]
'''
def errorCorrectionBasedOnEdits(examples, output, args):
    output.write("EDITs Correction... \n\t")
    #Detection
    true_positive = 0
    gold_standard_edits = 0#[]
    predictor_edits = 0#[]

    total = len(examples)
    for index in range(total):
        student_preposition = np.argmax(examples[index].original)
        correct_preposition = np.argmax(examples[index].correct)
        predictor_preposition = np.argmax(examples[index].predicted)
        #print (index)
        golden_edit = 0
        predictor_edit = 0
        '''if student_preposition >= args.class_limit:
            continue'''

        if student_preposition != correct_preposition:
            #gold_standard_edits.append(Edit(index, np.argmax(correct_preps)))
            gold_standard_edits = gold_standard_edits + 1
            golden_edit = 1
        if student_preposition != predictor_preposition:
            #predictor_edits.append(Edit(index, np.argmax(predictor_preps)))
            predictor_edits = predictor_edits + 1
            predictor_edit = 1
        if (predictor_edit == 1 and golden_edit == 1) and (predictor_preposition == correct_preposition):
            true_positive = true_positive + 1
    if gold_standard_edits > 0:
        recall = true_positive/gold_standard_edits
    else:
        recall = 0
    if predictor_edits > 0:
        precision = true_positive/predictor_edits
    else:
        precision = 0
    if recall + precision > 0:
        f1 = (2*recall*precision)/(recall+precision)
    else:
        f1 = 0
    if (recall + (math.pow(0.5, 2)*precision)) > 0:
        f05 = (1 + math.pow(0.5, 2))*(recall*precision)/(recall + (math.pow(0.5, 2)*precision))
    else:
        f05 = 0

    output.write("Correctly identified   " + str(true_positive) + "\n\t")
    output.write("Gold Standard Edit   s" + str(gold_standard_edits) + "\n\t")
    output.write("Predictor Edits   " + str(predictor_edits) + "\n\t")
    output.write("Precision:    " + str(precision) + "\n\t")
    output.write("Recall:    " + str(recall) + "\n\t")
    output.write("F1:    " + str(f1) + "\n\t")
    output.write("F05:    " + str(f05) + "\n\t")

def getPrecision(TPs, FPs):
    if (TPs + FPs) > 0:
        precision = TPs/(TPs + FPs)
    else:
        precision = 0
    return precision

def getRecall(TPs, FNs):
    if (TPs + FNs) > 0:
        recall = TPs / (TPs + FNs)
    else:
        recall = 0
    return recall

def getF1(recall, precision):
    if (recall+precision) > 0:
        f1 = (2 * recall * precision) / (recall + precision)
    else:
        f1 = 0
    return f1

def getF05(recall, precision):
    if (recall + (math.pow(0.5, 2)*precision)) > 0:
        f05 = (1 + math.pow(0.5, 2)) * (recall * precision) / (recall + (math.pow(0.5, 2) * precision))
    else:
        f05 = 0
    return f05

def getFPR(FPs, TNs):
    if (FPs + TNs) > 0:
        fpr = FPs / (FPs + TNs)
    else:
        fpr = 0
    return fpr

def getAccuracy(TPs, FPs, TNs, FNs):
    if TPs + FPs + TNs + FNs <= 0:
        return 0
    return (TPs + TNs)/(TPs + FPs + TNs + FNs)

def metrics(TPs, FPs, TNs, FNs, output, prefix = None, args = None):
    precision = getPrecision(TPs, FPs)
    recall = getRecall(TPs, FNs)
    f1 = getF1(recall, precision)
    f05 = getF05(recall, precision)
    accuracy = getAccuracy(TPs, FPs, TNs, FNs)

    output.write("\tTP " + str(TPs) + "\n\t")
    output.write("TN " + str(TNs) + "\n\t")
    output.write("FP " + str(FPs) + "\n\t")
    output.write("FN " + str(FNs) + "\n\t")
    output.write ("Precision: " + str(precision*100) + "\n\t")
    output.write ("Recall: " + str(recall*100) + "\n\t")
    output.write ("F1: " + str(f1*100) + "\n\t")
    output.write ("F05: " + str(f05*100) + "\n\t")
    output.write ("Accuracy: " + str(accuracy*100) + "\n\t")
    if prefix != None and args != None:
        saveMetric(args.work_dir + "lists/" + prefix + "prec", precision, args)
        saveMetric(args.work_dir + "lists/" + prefix + "rec", recall, args)
        saveMetric(args.work_dir + "lists/" + prefix + "f1", f1, args)
        saveMetric(args.work_dir + "lists/" + prefix + "f05", f05, args)
        saveMetric(args.work_dir + "lists/" + prefix + "acc", accuracy, args)

def analyseClasses(examples, labels_per_class, predictions_per_class, auc, session, output, type_examples, args):
    for c in range(args.class_limit):
        labels = []
        predictions = []
        predictions_as_probabilities = []
        tp_threshold = [0] * 100
        tn_threshold = [0] * 100
        fp_threshold = [0] * 100
        fn_threshold = [0] * 100
        positive = [0] * 100
        negative = [0] * 100
        for ex in examples:
            probabilities = softmax(ex.predicted)
            predictions_as_probabilities.append(probabilities[c])
            if np.argmax(ex.predicted) == c:
                predictions.append(1)
            else:
                predictions.append(0)
            if np.argmax(ex.correct) == c:
                labels.append(1)
            else:
                labels.append(0)
            if np.argmax(ex.correct) == c and np.argmax(ex.predicted) == np.argmax(ex.correct):
                tp_threshold[int(probabilities[c]*99)] += 1
            if np.argmax(ex.correct) != c and np.argmax(ex.predicted) != c:
                tn_threshold[int(probabilities[c] * 99)] += 1
            if np.argmax(ex.correct) != c and np.argmax(ex.predicted) == c:
                fp_threshold[int(probabilities[c]*99)] += 1
            if np.argmax(ex.correct) == c and np.argmax(ex.predicted) != c:
                fn_threshold[int(probabilities[c] * 99)] += 1
            if np.argmax(ex.correct) == c:
                positive[int(probabilities[c] * 99)] += 1
            if np.argmax(ex.correct) != c:
                negative[int(probabilities[c] * 99)] += 1
        '''print("Class ", c)
        print("tp", tp_threshold)
        print("tn", tn_threshold)
        print("fp", fp_threshold)
        print("fn", fn_threshold)
        print("positive", positive)
        print("negative", negative)'''
        precision = []
        recall = []
        fpr = []
        f1 = []
        f05 = []
        accuracy = []
        for i in range(100):
            precision.append(getPrecision(np.sum(tp_threshold[i:100]), np.sum(fp_threshold[i:100])))
            recall.append(getRecall(np.sum(tp_threshold[i:100]), np.sum(fn_threshold[0:i+1])))
            fpr.append(getFPR(np.sum(fp_threshold[i:100]), np.sum(tn_threshold[0:i+1])))
            f1.append(getF1(precision[i], recall[i]))
            f05.append(getF1(precision[i], recall[i]))
            accuracy.append(getAccuracy(np.sum(tp_threshold[i:100]), np.sum(fp_threshold[i:100]), np.sum(tn_threshold[0:i+1]), np.sum(fn_threshold[0:i+1])))
        '''print("precision", precision)
        print("recall", recall)
        print("false positive rate", fpr)
        print("F1", f1)
        print("F05", f05)
        print("accuracy", accuracy)'''

        auc_data = {labels_per_class: labels, predictions_per_class: predictions_as_probabilities}
        auc_per_class = session.run(auc, feed_dict=auc_data)
        output.write("AUC for class " + str(c) + " is: " + str(auc_per_class) + "\t\n")
        output.write("Metrics for class: " + str(auc_per_class) + "\t\n")
        saveMetric(args.work_dir + "lists/" + type_examples + "pred_class_" + str(c), predictions_as_probabilities, args)
        saveMetric(args.work_dir + "lists/" + type_examples + "lab_class_" + str(c), labels, args)
        saveMetric(args.work_dir + "lists/" + type_examples + "auc_class_" + str(c), auc_per_class, args)

        saveMetric(args.work_dir + "lists/" + type_examples + "tp_class_" + str(c), tp_threshold, args)
        saveMetric(args.work_dir + "lists/" + type_examples + "tn_class_" + str(c), tn_threshold, args)
        saveMetric(args.work_dir + "lists/" + type_examples + "fp_class_" + str(c), fp_threshold, args)
        saveMetric(args.work_dir + "lists/" + type_examples + "fn" + str(c), fn_threshold, args)
        saveMetric(args.work_dir + "lists/" + type_examples + "ps_class_" + str(c), positive, args)
        saveMetric(args.work_dir + "lists/" + type_examples + "ng" + str(c), negative, args)

        saveMetric(args.work_dir + "lists/" + type_examples + "prec_class_" + str(c), precision, args)
        saveMetric(args.work_dir + "lists/" + type_examples + "rec_class_" + str(c), recall, args)
        saveMetric(args.work_dir + "lists/" + type_examples + "fpr_class_" + str(c), fpr, args)
        saveMetric(args.work_dir + "lists/" + type_examples + "f1_class_" + str(c), f1, args)
        saveMetric(args.work_dir + "lists/" + type_examples + "f05_class_" + str(c), f05, args)
        saveMetric(args.work_dir + "lists/" + type_examples + "accu_class_" + str(c), accuracy, args)
        metrics_per_class(labels, predictions, output)

def metrics_per_class(labels, predictions, output):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for l, p in zip(labels, predictions):
        if l == 1 and p == 1:
            tp += 1
        elif l == 1 and p == 0:
            fn += 1
        elif l == 0 and p == 1:
            fp += 1
        elif l == 0 and p == 0:
            tn += 1
    metrics(tp, fp, tn, fn, output)

def saveMetric(fname, object, args):
    saving_lists = file_io.FileIO(dp.createName(fname, args) + ".pickle",
                                  mode="w")
    pickle.dump(object, saving_lists, protocol=2)
    saving_lists.close()

def appliedPostAnalysis(examples, output, prefix, args):
    output.write("WAS Evaluation Schema Applying Post-Precessing \n\t")
    # Detection
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total = len(examples)

    # DETECTION
    for i in range(total):
        student_preposition = np.argmax(examples[i].original)
        correct_preposition = np.argmax(examples[i].correct)
        predictor_preposition = np.argmax(examples[i].predicted)
        predictions = softmax(examples[i].predicted)

        #Analysis for top classes
        if predictor_preposition in (0, 5, 6, 7, 12):
            if predictions[predictor_preposition] <= 0.70:
                predictor_preposition = student_preposition


        #Seond level classes
        elif predictor_preposition in (1, 2, 3, 4, 8, 9, 15):
            if predictions[predictor_preposition] <= 0.90:
                predictor_preposition = student_preposition

        #last level classes
        elif predictor_preposition in (10, 11, 13, 14, 16, 17, 18, 19):
            if predictions[predictor_preposition] <= 1.0:
                predictor_preposition = student_preposition

        # Error is not flagged, and error does not exist
        if student_preposition == predictor_preposition and correct_preposition == predictor_preposition:
            true_negative = true_negative + 1
        # Flags a possible error, but there is not error or correction is wrong
        elif student_preposition != predictor_preposition and correct_preposition == student_preposition:
            false_positive = false_positive + 1
        # Does not flag error, but error exists
        elif student_preposition == predictor_preposition and correct_preposition != predictor_preposition:
            false_negative = false_negative + 1
        # Flags a possible error, and correction is right
        elif student_preposition != predictor_preposition and correct_preposition == predictor_preposition:
            true_positive = true_positive + 1
        # flag an error but currection is wrong
        elif student_preposition != predictor_preposition and correct_preposition != predictor_preposition:
            false_negative += 1
            false_positive += 1
    metrics(true_positive, false_positive, true_negative, false_negative, output, prefix + "final", args)

def appliedPostAnalysisDetection(examples, output, prefix, args):
    output.write("Detection Applying Post-Precessing \n\t")
    # Detection
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total = len(examples)

    # DETECTION
    for i in range(total):
        student_preposition = np.argmax(examples[i].original)
        correct_preposition = np.argmax(examples[i].correct)
        predictor_preposition = np.argmax(examples[i].predicted)
        predictions = softmax(examples[i].predicted)

        #Analysis for top classes
        if predictor_preposition in (0, 5, 6, 7, 12):
            if predictions[predictor_preposition] <= 0.70:
                predictor_preposition = student_preposition


        #Seond level classes
        elif predictor_preposition in (1, 2, 3, 4, 8, 9, 15):
            if predictions[predictor_preposition] <= 0.90:
                predictor_preposition = student_preposition

        #last level classes
        elif predictor_preposition in (10, 11, 13, 14, 16, 17, 18, 19):
            if predictions[predictor_preposition] <= 1.0:
                predictor_preposition = student_preposition

        # Error is not flagged, and error does not exist
        if student_preposition == predictor_preposition and correct_preposition == predictor_preposition:
            true_negative = true_negative + 1
        # Flags a possible error, but there is not error or correction is wrong
        elif student_preposition != predictor_preposition and correct_preposition == student_preposition:
            false_positive = false_positive + 1
        # Does not flag error, but error exists
        elif student_preposition == predictor_preposition and correct_preposition != predictor_preposition:
            false_negative = false_negative + 1
        # Flags a possible error, and correction is right
        elif student_preposition != predictor_preposition and correct_preposition == predictor_preposition:
            true_positive = true_positive + 1
        elif student_preposition != predictor_preposition and correct_preposition != predictor_preposition:
            true_positive += 1
    metrics(true_positive, false_positive, true_negative, false_negative, output, prefix + "det", args)