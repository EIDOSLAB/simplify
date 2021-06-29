import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """
    Evaluates model Top-k accuracy.
    :param output: PyTorch model's output.
    :param target: Inference ground truth.
    :param topk: Tuple defining the Top-k accuracy to be evaluated e.g. (1, 5) equals to Top-1 and Top-5 accuracy
    :return: List containing the Top-K accuracy values.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def evaluate_jaccard(outputs, targets):
    eps = 1e-15
    jaccard_targets = (targets == 1).float()
    jaccard_outputs = torch.sigmoid(outputs)

    intersection = (jaccard_outputs * jaccard_targets).sum()
    union = jaccard_outputs.sum() + jaccard_targets.sum()

    jaccard = (intersection + eps) / (union - intersection + eps)

    return jaccard


def evaluate_dice(jaccard):
    return 2 * jaccard / (1 + jaccard)


@torch.no_grad()
def test_model(model, loss_function, dataloader, device, task, amp=False, desc=None):
    """
    Evaluates PyTorch model performance.
    :param model: PyTorch model to evaluate.
    :param loss_function: Loss function used to evaluate the model Loss.
    :param dataloader: DataLoader on which evaluate the performance.
    :param device: Device on which to map the data, cpu or cuda:x where x is the cuda id.
    :return: Top-1 accuracy, Top-5 accuracy and Loss.
    """
    losses = AverageMeter('Loss')
    measure_1 = AverageMeter('@1Accuracy')
    measure_2 = AverageMeter('@5Accuracy')

    if dataloader is not None:
        model.eval()

        if desc is not None:
            pbar = tqdm(dataloader, total=len(dataloader))
            pbar.set_description(desc)
        else:
            pbar = dataloader

        for data, target in pbar:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if amp:
                with autocast():
                    output = model(data)
                    loss = loss_function(output, target)
            else:
                output = model(data)
                loss = loss_function(output, target)

            losses.update(loss.item(), data.size(0))

            if task == "classification":
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                measure_1.update(acc1[0], data.size(0))
                measure_2.update(acc5[0], data.size(0))
            elif task == "segmentation":
                jaccard = evaluate_jaccard(output, target)
                dice = evaluate_dice(jaccard.item())
                measure_1.update(jaccard, data.size(0))
                measure_2.update(dice, data.size(0))

        model.train()

    return measure_1.avg, measure_2.avg, losses.avg
