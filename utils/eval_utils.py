import torch
import logging
import numpy as np
from typing import Union
from datasets.imagenet_subsets import IMAGENET_D_MAPPING
from tqdm import tqdm


logger = logging.getLogger(__name__)


def split_results_by_domain(domain_dict: dict, data: list, predictions: torch.tensor):
    """
    Separates the label prediction pairs by domain
    Input:
        domain_dict: Dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
        data: List containing [images, labels, domains, ...]
        predictions: Tensor containing the predictions of the model
    Returns:
        domain_dict: Updated dictionary containing the domain seperated label prediction pairs
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict


def eval_domain_dict(domain_dict: dict, domain_seq: list):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    Input:
        domain_dict: Dictionary containing the labels and predictions for each domain
        domain_seq: Order to print the results (if all domains are contained in the domain dict)
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    domain_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting the results by domain...")
    for key in domain_names:
        label_prediction_arr = np.array(domain_dict[key])  # rows: samples, cols: (label, prediction)
        correct.append((label_prediction_arr[:, 0] == label_prediction_arr[:, 1]).sum())
        num_samples.append(label_prediction_arr.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")


def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 print_every: int,
                 device: Union[str, torch.device]):

    num_correct = 0
    num_samples = 0
    num_correct_stu_set = 0
    num_stu_set = 0

    num_zeros = 0


    Ineffs = []
    max_scores = []
    accs = []
    diffs = []

    entropys = []
    is_accs = []
    is_errs = []
    nlls = []

    
    with torch.no_grad():
        num_iter = len(data_loader)
        pbar = tqdm(range(num_iter))
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]

            data = data[:2]  # remove labels
            
            # output, losses, stu_pred_set = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            output, losses, stu_pred_set = model([da.to(device) for da in data]) if isinstance(data, list) else model(data.to(device))
            predictions = output.argmax(1)

            # 统计每一个样本的熵和预测是否正确
            entropy = -torch.sum(output.softmax(1) * output.log_softmax(1), 1)
            entropys.append(entropy.cpu().numpy())
            is_accs.append((predictions == labels.to(device)).cpu().numpy())
            is_errs.append((predictions != labels.to(device)).cpu().numpy())

            nlls.extend(-torch.log(output.softmax(1)[range(len(labels)), labels.to(device).long()]).cpu().numpy())
                           
            max_scores.append(output.softmax(1).max(1)[0].cpu().numpy())
            Ineffs.append(stu_pred_set.sum(1).cpu().numpy().mean())

            pbar.set_description(f"losses: {', '.join([f'{k}: {v:.3f}' for k, v in losses.items()])}", refresh=True)
            pbar.update()

            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)


            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)
            
    
            # track progress
            num_samples += imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
            num_correct += (predictions == labels.to(device)).float().sum()
            num_correct_stu_set += (stu_pred_set[range(len(stu_pred_set)), labels.to(device).long()].sum()).float()

            num_stu_set += stu_pred_set.sum().float()

            # 统计pred_set全为0的样本数
            num_zeros += (stu_pred_set.sum(1) == 0).sum().item()

            accs.append((predictions == labels.to(device)).float().sum().item() / len(labels))


            if print_every > 0 and (i+1) % print_every == 0:
                logger.info(f"#batches={i+1:<6} #samples={num_samples:<9} error = {1 - num_correct / num_samples:.2%}")

            if dataset_name == "ccc" and num_samples >= 7500000:
                break
        pbar.close()

    
    # 计算熵和预测是否正确的相关性，绘制AUROC曲线
    entropys = np.concatenate(entropys)
    max_scores = np.concatenate(max_scores)
    is_accs = np.concatenate(is_accs)
    is_errs = np.concatenate(is_errs)
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr_entropy, tpr_entropy, thresholds_entropy = roc_curve(is_errs, entropys)
    auroc_entropy = roc_auc_score(is_errs, entropys)
    logger.info(f"AUROC_entropy: {auroc_entropy:.4f}")

    fpr_max_scores, tpr_max_scores, thresholds_max_scores = roc_curve(is_errs, - max_scores)
    auroc_max_scores = roc_auc_score(is_errs, - max_scores)
    logger.info(f"AUROC_max-score: {auroc_max_scores:.4f}")

    # 绘制AUROC曲线(entropy和max_scores的画到同一张图中)
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(fpr_entropy, tpr_entropy, label=f'entropy (auroc = {auroc_entropy:.4f})')
    # plt.plot(fpr_max_scores, tpr_max_scores, label=f'max_scores (auroc = {auroc_max_scores:.4f})')
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.savefig('result/roc_curve.png')
    # plt.close()



    accuracy = num_correct.item() / num_samples
    cover_stu_set = num_correct_stu_set.item() / num_samples
    num_avg_stu_set = num_stu_set.item() / num_samples

   
    return accuracy, domain_dict, num_samples, cover_stu_set, num_avg_stu_set, max_scores, accs, Ineffs, is_accs, nlls



def get_features(model: torch.nn.Module, 
                 data_loader: torch.utils.data.DataLoader, 
                 device: Union[str, torch.device]):
    
    features = torch.tensor([])
    labels = torch.tensor([])
    predictions = torch.tensor([])
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            imgs, label = data[0], data[1]
            feature,logit = model.forward_get_features(imgs.to(device))
            feature = feature.to('cpu')
            logit = logit.to('cpu')
            features = torch.cat((features, feature), dim=0)
            labels = torch.cat((labels, label), dim=0)
            prediction = logit.argmax(1)
            predictions = torch.cat((predictions, prediction), dim=0)

    return features, labels, predictions
    



def get_accuracy_tsne(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 print_every: int,
                 device: Union[str, torch.device]):

    num_correct = 0.
    num_samples = 0
    outputs_features = []
    outputs_features_ema = []
    outputs_features_anchor = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(data_loader), total=len(data_loader)):
            imgs, labels = data[0], data[1]
            output,outputs_feature, outputs_feature_ema, outputs_feature_anchor = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device), need_feature=True)

            # 根据labels筛选出类别为1的输出
            outputs_feature = outputs_feature[labels == 4]
            outputs_feature_ema = outputs_feature_ema[labels == 4]
            outputs_feature_anchor = outputs_feature_anchor[labels == 4]

            outputs_features.append(outputs_feature)
            outputs_features_ema.append(outputs_feature_ema)
            outputs_features_anchor.append(outputs_feature_anchor)

            predictions = output.argmax(1)

            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

            num_correct += (predictions == labels.to(device)).float().sum()

            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)

            # track progress
            num_samples += imgs[0].shape[0] if isinstance(imgs, list) else imgs.shape[0]
            if print_every > 0 and (i+1) % print_every == 0:
                logger.info(f"#batches={i+1:<6} #samples={num_samples:<9} error = {1 - num_correct / num_samples:.2%}")

            if dataset_name == "ccc" and num_samples >= 7500000:
                break

    outputs_features = torch.cat(outputs_features, dim=0)
    outputs_features_ema = torch.cat(outputs_features_ema, dim=0)
    outputs_features_anchor = torch.cat(outputs_features_anchor, dim=0)

    accuracy = num_correct.item() / num_samples
    return accuracy, domain_dict, num_samples, outputs_features, outputs_features_ema, outputs_features_anchor
