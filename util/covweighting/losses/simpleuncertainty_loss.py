import torch
from torch import nn

from .simplebase_loss import SimpleBaseLoss


class SimpleUncertaintyLoss(SimpleBaseLoss):

    """
        Wrapper of the BaseLoss which weighs the losses according to the Uncertainty Weighting
        method by Kendall et al. Now for 32 losses.
    """

    def __init__(self, args):
        super(SimpleUncertaintyLoss, self).__init__(args)
        # These are the log(sigma ** 2) parameters.
        self.log_vars = nn.Parameter(torch.zeros(self.num_losses, dtype=torch.float32), requires_grad=True)
        
        self.unweighted_l = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)
        self.weighted_l   = torch.zeros((self.num_losses,), requires_grad=False).type(torch.FloatTensor).to(self.device)

    def to_eval(self):
        self.alphas = torch.exp(-self.log_vars).detach().clone()
        self.train = False

    def to_train(self):
        self.train = True

    def forward(self, pred, target):
        unweighted_losses = super(SimpleUncertaintyLoss, self).forward(pred, target)
        # TODO: inserted the following line, subsequent codes seem to expected loss outputs
        unweighted_losses = [loss(pred, target) for loss in unweighted_losses] # [unweighted_losses[0](pred, target), unweighted_losses[1](pred, target)]
        # -- Kendall's Gaussian method: L_total = sum_i exp(-s_i) * L_i + s_i --
        losses = [torch.exp(-self.log_vars[i]) * loss + self.log_vars[i] for i, loss in enumerate(unweighted_losses)]

        self.unweighted_l = unweighted_losses
        self.weighted_l   = losses
        loss = sum(losses)
        return loss
