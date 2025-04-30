import csv
import torch
import sys
sys.path.append('C:\\Users\\fly\\Desktop\\')
from models.TextCNN import TextCNN
from losses import FocalDiceLoss,ResampleLoss,gain_class_freq_and_train_num
from DataLoad import data_load
from train import DataTrain, predict_with_thresholds, CosineScheduler,create_logger,optimize_thresholds_with_freq,evaluate_with_thresholds
torch.manual_seed(20230226)  # Set the random seed for CPU
torch.cuda.manual_seed(20230226)  # Set the random seed for GPU
torch.backends.cudnn.deterministic = True  # Fix the GPU computation method
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select device
# Types of therapeutic peptides in the dataset
RMs = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP',
       'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP',
       'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']

# Titles in the CSV for saving running results
title1 = ['Model', "Loss", 'Aiming', 'Coverage', 'Accuracy',
          'Absolute_True', 'Absolute_False', 'RunTime']

def spent_time(start, end):
    # Calculate the running time of the code
    epoch_time = end - start
    minute = int(epoch_time / 60)  # minutes
    secs = int(epoch_time - minute * 60)  # seconds
    return minute, secs


def save_results(model_name, loss_name, start, end, test_score, class_scores, file_path):
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    content = [[model_name, loss_name,
                '%.3f' % test_score[0],
                '%.3f' % test_score[1],
                '%.3f' % test_score[2],
                '%.3f' % test_score[3],
                '%.3f' % test_score[4],
                '%.3f' % (end - start),
                now]]

    class_title = ['Class', 'TP', 'FP', 'TN', 'FN']
    class_results = []

    for i, rm in enumerate(RMs):
        class_results.append([rm, class_scores[0][i], class_scores[1][i], class_scores[2][i], class_scores[3][i]])

    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path, encoding='gbk')
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerows(content)
                writer.writerow([])
                writer.writerow(['Model'] + class_title)
                writer.writerows(class_results)

        except Exception as e:
            print(f"Error reading existing CSV file: {e}. Creating new file.")
            with open(file_path, 'w', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(
                    ['Model', 'Loss', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'Runtime',
                     'Timestamp'])
                writer.writerows(content)
                writer.writerow([])  # Add an empty row for separation
                writer.writerow(['Model'] + class_title)  # Use consistent header
                writer.writerows(class_results)
    else:
        with open(file_path, 'w', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(
                ['Model', 'Loss', 'Aiming', 'Coverage', 'Accuracy', 'Absolute_True', 'Absolute_False', 'Runtime',
                 'Timestamp'])
            writer.writerows(content)
            writer.writerow([])
            writer.writerow(['Model'] + class_title)
            writer.writerows(class_results)

import torch
import time
import os
import pandas as pd
def TrainAndTest(args):
    logger = create_logger()
    logger.info(f"This task is {args.task}")
    models_file = f'result/{args.task}_models.txt'
    Time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    parse_file = f"result/{args.task}_pares.txt"
    with open(parse_file, 'a') as file1:
        file1.write(Time + '\n')
        print(args, file=file1)
        file1.write('\n')

    file_path = f'result/model_select.csv'
    logger.info("Data is loading ......（￣︶￣）↗　")

    train_datasets, test_datasets, subtests, weight = data_load(
        batch=args.batch_size,
        train_direction=args.train_direction,
        test_direction=args.test_direction,
        subtest=args.subtest,
        CV=False
    )
    logger.info("Data is loaded!ヾ(≧▽≦*)o")
    class_freq, train_num = gain_class_freq_and_train_num()
    start_time = time.time()
    for i in range(len(train_datasets)):
        train_dataset = train_datasets[i]
        test_dataset = test_datasets[i]
        train_start = time.time()

        # Initialize model
        if args.model == 'TextCNN':
            model = TextCNN(args.vocab_size, args.embedding_size, args.filter_num,
                            args.filter_size, args.output_size, args.dropout)
        else:
            raise Exception('Unexpected model {}'.format(args.model))

        model_name = model.__class__.__name__
        title_task = f"{args.task}+{model_name}"

        # Record model structure
        model_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        with open(models_file, 'a') as file2:
            file2.write(model_time + '\n')
            print(model, file=file2)
            file2.write('\n')
        logger.info(f"{model_name} is loading from {args.model_path}......")
        try:
            model.load_state_dict(torch.load(args.model_path))
            model = model.to(DEVICE)
            loss_name = "FocalDiceLoss+ResampleLoss"
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        # Optimize thresholds
        logger.info("Optimizing thresholds...")
        optimal_thresholds = optimize_thresholds_with_freq(
            model,
            train_dataset,
            class_freq,
            train_num,
            device=DEVICE
        )
        # Use optimized thresholds for prediction and evaluation
        model_predictions, true_labels = predict_with_thresholds(
            model,
            test_dataset,
            thresholds=optimal_thresholds,
            device=DEVICE
        )
        # Evaluate results
        test_score = evaluate_with_thresholds(model_predictions, true_labels)

        # Get TP, FP, TN, FN for each class
        class_scores = get_per_class_metrics(model_predictions, true_labels)

        # Save results
        test_end = time.time()
        save_results(title_task, loss_name, train_start, test_end, test_score, class_scores, file_path)
        # Print evaluation results
        run_time = time.time()
        m, s = spent_time(start_time, run_time)
        logger.info(f"{args.task}, {model_name}'s runtime:{m}m{s}s")
        logger.info("Test Results:")
        logger.info(f'Aiming: {test_score[0]:.3f}')
        logger.info(f'Coverage: {test_score[1]:.3f}')
        logger.info(f'Accuracy: {test_score[2]:.3f}')
        logger.info(f'Absolute True: {test_score[3]:.3f}')
        logger.info(f'Absolute False: {test_score[4]:.3f}\n')


def get_per_class_metrics(predictions, true_labels):
    """
    Calculate TP, FP, TN, FN for each class

    Parameters:
    predictions: Model prediction results (after binarization)
    true_labels: True labels

    Returns:
    class_scores: List containing TP, FP, TN, FN in format [TP_list, FP_list, TN_list, FN_list]
    """
    num_classes = predictions.shape[1]
    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []

    for i in range(num_classes):
        TP = ((predictions[:, i] == 1) & (true_labels[:, i] == 1)).sum().item()
        FP = ((predictions[:, i] == 1) & (true_labels[:, i] == 0)).sum().item()
        TN = ((predictions[:, i] == 0) & (true_labels[:, i] == 0)).sum().item()
        FN = ((predictions[:, i] == 0) & (true_labels[:, i] == 1)).sum().item()

        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)

    return [TP_list, FP_list, TN_list, FN_list]