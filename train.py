from losses import MultiLabelContrastiveLoss
import time
import torch
import math
import numpy as np
import estimate
from torch.optim import lr_scheduler
import sys
sys.path.append('C:\\Users\\fly\\Desktop\\')
import logging
def create_logger(log_name='test.log', mode='a'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # Create file handler and set level
    file_handler = logging.FileHandler(log_name, mode=mode)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


class DataTrain:
    def __init__(self, model, optimizer, criterion_list, scheduler=None, loss_weights=None,
                 device="cuda", flooding_level=0.01):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion_list = criterion_list
        self.loss_weights = loss_weights
        self.lr_scheduler = scheduler
        self.device = device
        self.flooding_level = flooding_level
        self.contrastive_criterion = MultiLabelContrastiveLoss().to(device)

    def train_step(self, train_iter, test_iter=None, epochs=None, model_name='', va=True):
        steps = 1
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            train_loss = []

            for train_data, train_length, train_label in train_iter:
                self.model.train()
                train_data = train_data.to(self.device)
                train_length = train_length.to(self.device)
                train_label = train_label.to(self.device)

                result, features = self.model(train_data, return_features=True)
                contrast_loss = self.contrastive_criterion(features, train_label)

                loss1 = self.criterion_list[0](result, train_label.float())
                loss2 = self.criterion_list[1](result, train_label.float())

                loss = self.loss_weights[0] * loss1 + self.loss_weights[1] * loss2 + 0.3 * contrast_loss
                loss = (loss - self.flooding_level).abs() + self.flooding_level

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        self.lr_scheduler.step()
                    else:
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)
            steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time
            print(f'{model_name}|Epoch:{epoch:003}/{epochs}|Run time:{epoch_time:.2f}s')
            print(f'Train loss:{np.mean(train_loss):.4f}')


def custom_sigmoid_multi_threshold(x, thresholds, k=10):    #k=1
    """Sigmoid function supporting multiple thresholds"""
    thresholds = thresholds.view(1, -1)
    return 1 / (1 + torch.exp(-k * (x - thresholds)))

def predict_with_thresholds(model, data, thresholds, device="cuda", return_features=False):
    model.to(device)
    model.eval()
    all_predictions = []
    all_labels = []
    all_features = []

    thresholds = torch.tensor(thresholds).to(device)

    with torch.no_grad():
        for batch in data:
            x, l, y = batch
            x = x.to(device)

            if return_features:
                scores, features = model(x, return_features=True)
                all_features.extend(features.cpu().numpy())
            else:
                scores = model(x, return_features=False)

            # Use sigmoid function
            sigmoid_scores = custom_sigmoid_multi_threshold(scores, thresholds)
            predictions = torch.zeros_like(scores)
            for i in range(scores.shape[1]):
                # Compare sigmoid scores with thresholds
                predictions[:, i] = (sigmoid_scores[:, i] > thresholds[i]).float()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)

def evaluate_with_thresholds(predictions, labels):
    """Evaluate binarized prediction results"""
    absolute_true = 0
    total_samples = len(predictions)

    # Calculate the number of perfectly matched samples
    for i in range(total_samples):
        if np.array_equal(predictions[i], labels[i]):
            absolute_true += 1
    absolute_true_rate = absolute_true / total_samples
    other_metrics = list(estimate.evaluate(predictions, labels, threshold=None))
    metrics = (
        other_metrics[0],  # aiming
        other_metrics[1],  # coverage
        other_metrics[2],  # accuracy
        absolute_true_rate,  #absolute_true
        other_metrics[3]  # absolute_false
    )
    return metrics
def optimize_thresholds_with_freq(model, val_data, class_freq, train_num, device="cuda", init_thresholds=None):
    """Threshold optimization considering class frequency"""
    init_thresholds = [0] * 21
    predictions = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for batch in val_data:
            x, l, y = batch
            x = x.to(device)
            score = model(x, return_features=False)
            predictions.append(score.cpu())
            true_labels.append(y.cpu())

    predictions = torch.cat(predictions, dim=0).numpy()
    true_labels = torch.cat(true_labels, dim=0).numpy()

    class_ratios = [freq / train_num for freq in class_freq]
    optimal_thresholds = []

    # Optimize threshold for each class
    for i in range(21):
        ratio = class_ratios[i]
        class_preds = predictions[:, i]
        class_labels = true_labels[:, i]

        # Set search range
        if ratio < 0.005:
            threshold_range = np.arange(0.1, 0.3, 0.01)
        elif ratio < 0.02:
            threshold_range = np.arange(0.3, 0.5, 0.01)
        elif ratio < 0.05:
            threshold_range = np.arange(0.4, 0.6, 0.01)
        elif ratio < 0.08:
            threshold_range = np.arange(0.45, 0.65, 0.01)
        elif ratio < 0.15:
            threshold_range = np.arange(0.5, 0.7, 0.01)
        else:
            threshold_range = np.arange(0.55, 0.8, 0.01)

        best_threshold = init_thresholds[i]
        best_score = float('-inf')

        # Try different thresholds
        for threshold in threshold_range:
            pred_labels = (class_preds > threshold).astype(float)
            current_scores = estimate.evaluate(
                pred_labels.reshape(1, -1),
                class_labels.reshape(1, -1),
                threshold=None
            )
            # Adjust scoring weights based on class frequency
            if ratio < 0.05:
                combined_score = (
                        current_scores[0] * 0.3 +
                        current_scores[1] * 0.1 +
                        current_scores[2] * 0.6
                )
            else:
                combined_score = (
                        current_scores[0] * 0.4 +
                        current_scores[1] *0.3 +
                        current_scores[2] * 0.3
                )

            if combined_score > best_score:
                best_score = combined_score
                best_threshold = threshold

        optimal_thresholds.append(best_threshold)
        print(f"Class {i}: ratio = {ratio:.4f}, threshold = {best_threshold:.3f}")

    return optimal_thresholds


class CosineScheduler:
    # Learning rate decay
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        # Initialization function, setting initial learning rate, final learning rate, warmup steps, etc.
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps  # Calculate maximum steps

    def get_warmup_lr(self, epoch):
        # Return learning rate during warmup phase
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch - 1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase # Return current learning rate in warmup phase


    def __call__(self, epoch):
        # Calculate learning rate when the object is called
        if epoch < self.warmup_steps:# If current epoch is less than warmup steps
            return self.get_warmup_lr(epoch) # Return learning rate during warmup phase
        if epoch <= self.max_update:# If current epoch is less than or equal to maximum update steps
            # Calculate learning rate using cosine annealing function

            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch - 1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr # Return current learning rate