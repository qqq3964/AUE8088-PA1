from torchmetrics import Metric
import torch

# [TODO] Implement this!
class MyF1Score(Metric):
    def __init__(self):
        super().__init__()
        num_classes = 200
        self.add_state('TP', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('FP', default=torch.zeros(num_classes), dist_reduce_fx='sum')
        self.add_state('FN', default=torch.zeros(num_classes), dist_reduce_fx='sum')
    
    def update(self, preds, target):
        # B, clasees
        preds_ = torch.zeros_like(preds)
        preds_idx = torch.argmax(preds, dim=1)
        preds_[torch.arange(preds_.shape[0]), preds_idx] = 1
        
        # B, classse
        target_ = torch.zeros_like(preds)
        target_[torch.arange(target.shape[0]), target] = 1
        
        # True Positive
        self.TP += torch.sum(preds_ * target_, dim=0)
        
        # False Positive
        self.FP += torch.sum(preds_ * (1 - target_), dim=0)
        
        # False Negative
        self.FN += torch.sum((1 - preds_) * target_, dim=0)
        
    def compute(self):
        # precision and recall
        eps = 1e-8
        precision = self.TP / (self.TP + self.FP + eps)
        recall = self.TP / (self.TP + self.FN + eps)
        f1_score = (2 * precision * recall) / (precision + recall + eps)
        
        # macro f1
        return f1_score.mean()

class MyAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds, target):
        # [TODO] The preds (B x C tensor), so take argmax to get index with highest confidence
        preds = torch.argmax(preds, dim=1)
        
        # [TODO] check if preds and target have equal shape
        if preds.shape != target.shape:
            raise Exception("shape is not same each other!")

        # [TODO] Count the number of correct prediction
        correct = torch.sum(preds == target)

        # Accumulate to self.correct
        self.correct += correct

        # Count the number of elements in target
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total.float()
