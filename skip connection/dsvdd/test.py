import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from torchsummary import summary
from torchmetrics.functional import auroc

def eval(net, c, dataloader, device):
    """Testing the Deep SVDD model"""

    scores = []
    labels = []
    new_scores = np.zeros(shape=(len(dataloader) * 50, 2))
    step = 0
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            z = net(x)
            score = torch.sum((z - c) ** 2, dim=1)
            
            for i in range(len(score)):
                point = int(score[i].cpu().item())
                label = int(y[i].item())
                new_scores[step] = [label, int(point < 26.05)]
                step += 1
            scores.append(score.detach().cpu())
            labels.append(y.cpu())

    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    # new_scores = torch.cat(new_scores).numpy()
    # accuracy = accuracy_score(new_scores[:, 0], new_scores[:, 1])
    precision, recall, fscore, _ = precision_recall_fscore_support(new_scores[:, 0], new_scores[:, 1])
    # print(f'accuracy - {accuracy}')
    print(f'precision - {precision}, recall - {recall}, fscore - {fscore}')
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(new_scores[:, 0], new_scores[:, 1])*100))
    print('AUROC score: {:.2f}'.format(auroc(torch.as_tensor(new_scores[:,0]), torch.as_tensor(new_scores[:,1]).int())*100))
    return labels, scores
