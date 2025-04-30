import torch.nn as nn
class FocalDiceLoss(nn.Module):
    """ Multi-label focal-dice loss """

    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.7, clip_neg=0.5, pos_weight=0.3, reduction='mean'):
        super(FocalDiceLoss, self).__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.reduction = reduction
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight

    def forward(self, input, target):
        assert input.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = nn.Sigmoid()(input)
        # predict = input
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        xs_pos = predict
        p_pos = predict
        if self.clip_pos is not None and self.clip_pos >= 0:
            m_pos = (xs_pos + self.clip_pos).clamp(max=1)
            p_pos = torch.mul(m_pos, xs_pos)
        num_pos = torch.sum(torch.mul(p_pos, target), dim=1)  # dim=1 sum by row
        den_pos = torch.sum(p_pos.pow(self.p_pos) + target.pow(self.p_pos), dim=1)

        xs_neg = 1 - predict
        p_neg = 1 - predict
        if self.clip_neg is not None and self.clip_neg >= 0:
            m_neg = (xs_neg + self.clip_neg).clamp(max=1)
            p_neg = torch.mul(m_neg, xs_neg)
        num_neg = torch.sum(torch.mul(p_neg, (1 - target)), dim=1)
        den_neg = torch.sum(p_neg.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1)

        loss_pos = 1 - (2 * num_pos) / den_pos
        loss_neg = 1 - (2 * num_neg) / den_neg
        loss = loss_pos * self.pos_weight + loss_neg * (1 - self.pos_weight)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class MultiLabelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, pos_threshold=0.6, neg_threshold=0.1,
                 hard_mining_ratio=0.2, semi_hard_margin=0.1):
        super().__init__()
        self.temperature = temperature
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.hard_mining_ratio = hard_mining_ratio  # Hard sample ratio
        self.semi_hard_margin = semi_hard_margin  # Semi-hard sample margin

    def get_hard_negative_samples(self, sim_matrix, neg_mask):
        """Get hard negative samples"""
        if neg_mask.sum() == 0:
            return torch.tensor(0.0, device=sim_matrix.device)

        neg_sim = sim_matrix[neg_mask]
        # Select top k negative samples with highest similarity
        k = int(neg_sim.size(0) * self.hard_mining_ratio)
        k = max(1, min(k, neg_sim.size(0)))  # Ensure k is at least 1 and not exceeding total samples
        hard_negative_sim = torch.topk(neg_sim, k=k).values
        return hard_negative_sim

    def get_hard_positive_samples(self, sim_matrix, pos_mask):
        """Get hard positive samples"""
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=sim_matrix.device)

        pos_sim = sim_matrix[pos_mask]
        # Select bottom k positive samples with lowest similarity
        k = int(pos_sim.size(0) * self.hard_mining_ratio)
        k = max(1, min(k, pos_sim.size(0)))  # Ensure k is at least 1 and not exceeding total samples
        hard_positive_sim = torch.topk(pos_sim, k=k, largest=False).values
        return hard_positive_sim

    def get_semi_hard_negative_samples(self, sim_matrix, pos_mask, neg_mask):
        """Get semi-hard negative samples"""
        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=sim_matrix.device)

        # Get minimum similarity of positive pairs
        pos_sim_min = sim_matrix[pos_mask].min()
        neg_sim = sim_matrix[neg_mask]

        # Select negative samples with similarity in range [pos_sim_min - margin, pos_sim_min]
        semi_hard_mask = (neg_sim > (pos_sim_min - self.semi_hard_margin)) & (neg_sim < pos_sim_min)
        semi_hard_sim = neg_sim[semi_hard_mask]

        if semi_hard_sim.size(0) == 0:
            return self.get_hard_negative_samples(sim_matrix, neg_mask)
        return semi_hard_sim

    def forward(self, features, labels):
        labels = labels.float()
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]

        # Calculate label Jaccard similarity
        label_sim = torch.matmul(labels, labels.T)
        label_sum = labels.sum(1, keepdim=True) + labels.sum(1, keepdim=True).T
        label_sim = label_sim / (label_sum - label_sim + 1e-6)

        # Feature similarity
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        # Numerical stability processing
        sim_matrix = torch.clamp(sim_matrix, min=-50, max=50)

        eye_mask = ~torch.eye(batch_size, device=features.device, dtype=torch.bool)

        # Calculate positive sample mask and negative sample mask
        pos_mask = (label_sim > self.pos_threshold) & eye_mask
        neg_mask = (label_sim < self.neg_threshold) & eye_mask

        # Calculate hard positive sample loss
        pos_loss = 0
        if pos_mask.sum() > 0:
            hard_positive_sim = self.get_hard_positive_samples(sim_matrix, pos_mask)
            pos_loss = torch.log(1 + torch.exp(-hard_positive_sim)).mean()

        # Calculate hard negative sample loss (including semi-hard samples)
        neg_loss = 0
        if neg_mask.sum() > 0:
            # Get semi-hard negative samples
            semi_hard_sim = self.get_semi_hard_negative_samples(sim_matrix, pos_mask, neg_mask)
            if semi_hard_sim.numel() > 0:
                neg_loss = torch.log(1 + torch.exp(semi_hard_sim)).mean()
            else:
                # If no suitable semi-hard samples, use hard negative samples
                hard_negative_sim = self.get_hard_negative_samples(sim_matrix, neg_mask)
                neg_loss = torch.log(1 + torch.exp(hard_negative_sim)).mean()

        # Sample count weighting
        num_pos = pos_mask.sum()
        num_neg = neg_mask.sum()
        if num_pos > 0 and num_neg > 0:
            pos_weight = num_neg / (num_pos + num_neg)
            neg_weight = num_pos / (num_pos + num_neg)
            total_loss = pos_weight * pos_loss + neg_weight * neg_loss
        else:
            total_loss = pos_loss + neg_loss

        return total_loss




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
Train_path = "dataset/augmented_train.txt"
class_freq = [0] * 21  # Initialize counts for each peptide
train_num=0
with open(Train_path, 'r') as file:
    for line in file:
        if line.startswith('>'):  # Assume each line with data starts with '>'
            train_num+=1
            line = line.strip()[1:]  # Remove the '>' and any whitespace
            for i, char in enumerate(line):
                if char == '1':
                    class_freq[i] += 1

def gain_class_freq_and_train_num():
    return class_freq,train_num
tail_indexes = [i for i, v in enumerate(class_freq) if v < 500]
head_indexes = [i for i, v in enumerate(class_freq) if v >= 500]
class ResampleLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True, partial=False,
                 loss_weight=1.0, reduction='mean',
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None,  # None, 'by_instance', 'by_batch'
                 focal=dict(
                     focal=True,
                     alpha=0.5,
                     gamma=2,
                 ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 class_freq=None,
                 train_num=None):
        super(ResampleLoss, self).__init__()

        assert (use_sigmoid is True) or (partial is False)
        self.use_sigmoid = use_sigmoid
        self.partial = partial
        self.loss_weight = loss_weight
        self.reduction = reduction
        if self.use_sigmoid:
            if self.partial:
                self.cls_criterion = partial_cross_entropy
            else:
                self.cls_criterion = binary_cross_entropy
        else:
            self.cls_criterion = cross_entropy

        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.alpha = focal['alpha']  # change to alpha

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        self.class_freq = torch.from_numpy(np.asarray(class_freq)).float().cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = train_num  # only used to be divided by class_freq
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 1) * init_bias  ########################## bug fixed https://github.com/wutong16/DistributionBalancedLoss/issues/8

        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        if self.focal:
            logpt = self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(-logpt)
            wtloss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            alpha_t = torch.where(label == 1, self.alpha, 1 - self.alpha)
            loss = alpha_t * ((1 - pt) ** self.gamma) * wtloss  ####################### balance_param should be a tensor
            loss = reduce_loss(loss, reduction)  ############################ add reduction
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None):
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale + logits * labels
            if weight is not None:
                weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum(gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if 'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, avg_n)).cuda()
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
                      (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, min_n)).cuda()
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss

def partial_cross_entropy(pred,
                          label,
                          weight=None,
                          reduction='mean',
                          avg_factor=None):
    if pred.dim() != label.dim():
        label, weight = _expand_binary_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    mask = label == -1
    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    if mask.sum() > 0:
        loss *= (1-mask).float()
        avg_factor = (1-mask).float().sum()

    # do the reduction for the weighted loss
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss

def _expand_binary_labels(labels, label_weights, label_channels):
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

def cross_entropy(pred, label, weight=None, reduction='mean', avg_factor=None):
    # element-wise losses
    if label.size(-1) != pred.size(0):
        label = _squeeze_binary_labels(label)

    loss = F.cross_entropy(pred, label, reduction='none')

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss

def _squeeze_binary_labels(label):
    if label.size(1) == 1:
        squeeze_label = label.view(len(label), -1)
    else:
        inds = torch.nonzero(label >= 1).squeeze()
        squeeze_label = inds[:,-1]
    return squeeze_label




if __name__ == "__main__":
    u = torch.Tensor([[[1.0, 0.0],
                       [0.0, 1.0],
                       [-1.0, 0.0],
                       [0.0, -1.0]],
                      [[1.0, 0.0],
                       [0.0, 1.0],
                       [-1.0, 0.0],
                       [0.0, -1.0]],
                      [[2.0, 0.0],
                       [0.0, 2.0],
                       [-2.0, 0.0],
                       [0.0, -4.0]]])

    v = torch.Tensor([[[0.0, 0.0],
                       [0.0, 2.0],
                       [-2.0, 0.0],
                       [0.0, -3.0]],
                      [[2.0, 0.0],
                       [0.0, 2.0],
                       [-2.0, 0.0],
                       [0.0, -4.0]],
                      [[1.0, 0.0],
                       [0.0, 1.0],
                       [-1.0, 0.0],
                       [0.0, -1.0]]])

    print("Input shape is (B,W,H):", u.shape, v.shape)