import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=256, device="cpu", alpha=0.1):
        super(CenterLoss, self).__init__()

        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.loss = nn.MSELoss()
        self.register_buffer("centers", torch.randn(self.num_classes, self.feat_dim))
        self.alpha = alpha

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        target_centers = self.centers[labels]
        return self.loss(x, target_centers)

    def update_centers(self, features, targets):
        with torch.no_grad():
            self.centers -= self.get_center_delta(features, targets, self.alpha)

    def get_center_delta(self, features, targets, alpha):
        # implementation equation (4) in the center-loss paper
        features = features.view(features.size(0), -1)
        targets, indices = torch.sort(targets)
        target_centers = self.centers[targets]
        features = features[indices]

        delta_centers = target_centers - features
        uni_targets, indices = torch.unique(
            targets.cpu(), sorted=True, return_inverse=True)

        uni_targets = uni_targets.to(self.device)
        indices = indices.to(self.device)

        delta_centers = torch.zeros(
            uni_targets.size(0), delta_centers.size(1)
        ).to(self.device).index_add_(0, indices, delta_centers)

        targets_repeat_num = uni_targets.size()[0]
        uni_targets_repeat_num = targets.size()[0]
        targets_repeat = targets.repeat(
            targets_repeat_num).view(targets_repeat_num, -1)
        uni_targets_repeat = uni_targets.unsqueeze(1).repeat(
            1, uni_targets_repeat_num)
        same_class_feature_count = torch.sum(targets_repeat == uni_targets_repeat, dim=1).float().unsqueeze(1)

        delta_centers = delta_centers / (same_class_feature_count + 1.0) * alpha
        result = torch.zeros_like(self.centers)
        result[uni_targets, :] = delta_centers

        return result


if __name__ == "__main__":
    l = CenterLoss()
    data = torch.randn((4, 256))
    label = torch.LongTensor([1, 1, 2, 3])
    l(data, label)
