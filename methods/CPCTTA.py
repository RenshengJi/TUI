from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import random

import PIL
import torchvision.transforms as transforms
from augmentations.transforms_cotta import get_tta_transforms
from time import time
import logging
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from models.model import split_up_model
from utils.losses import Entropy

import logging
logger = logging.getLogger(__name__)
import os
import tqdm


import torchvision.datasets as datasets

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from datasets.data_loading import get_source_loader

from torchcp.classification.scores import THR
from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores.base import BaseScore


@ADAPTATION_REGISTRY.register()
class CPCTTA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.model = model
        self.arch_name = cfg.MODEL.ARCH

        batch_size_src = cfg.TEST.BATCH_SIZE if cfg.TEST.BATCH_SIZE > 1 else cfg.TEST.WINDOW_LENGTH
        _, self.src_loader = get_source_loader(dataset_name=cfg.CORRUPTION.DATASET,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               preprocess=model.model_preprocess,
                                               data_root_dir=cfg.DATA_DIR,
                                               batch_size=batch_size_src,
                                               ckpt_path=cfg.MODEL.CKPT_PATH,
                                               num_samples=cfg.SOURCE.NUM_SAMPLES,
                                               percentage=cfg.SOURCE.PERCENTAGE,
                                               workers=min(cfg.SOURCE.NUM_WORKERS, os.cpu_count()))
        self.src_loader_iter = iter(self.src_loader)
        # define the prototype paths
        arch_name = cfg.MODEL.ARCH
        ckpt_path = cfg.MODEL.CKPT_PATH
        proto_dir_path = os.path.join(cfg.CKPT_DIR, "prototypes")
        if self.dataset_name == "domainnet126":
            fname = f"protos_{self.dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth"
        else:
            fname = f"protos_{self.dataset_name}_{arch_name}.pth"
        fname = os.path.join(proto_dir_path, fname)

        self.softmax_entropy = Entropy()
        self.feature_extractor, self.classifier = split_up_model(self.model, self.arch_name, self.dataset_name, split2=False)

        # get source prototypes
        if os.path.exists(fname):
            logger.info("Loading class-wise source prototypes...")
            self.prototypes_src = torch.load(fname)
        else:
            os.makedirs(proto_dir_path, exist_ok=True)
            features_src = torch.tensor([])
            labels_src = torch.tensor([])
            logger.info("Extracting source prototypes...")
            with torch.no_grad():
                for data in tqdm.tqdm(self.src_loader):
                    x, y = data[0], data[1]
                    tmp_features = self.feature_extractor(x.to(self.device))
                    features_src = torch.cat([features_src, tmp_features.view(tmp_features.shape[:2]).cpu()], dim=0)
                    labels_src = torch.cat([labels_src, y], dim=0)
                    if len(features_src) > 100000:
                        break

            # create class-wise source prototypes
            self.prototypes_src = torch.tensor([])
            for i in range(self.num_classes):
                mask = labels_src == i
                self.prototypes_src = torch.cat([self.prototypes_src, features_src[mask].mean(dim=0, keepdim=True)], dim=0)

            torch.save(self.prototypes_src, fname)

        self.prototypes_src = self.prototypes_src.to(self.device).unsqueeze(1)
        self.prototype_labels_src = torch.arange(start=0, end=self.num_classes, step=1).to(self.device).long()


        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.feature_extractor_src, self.classifier_src = split_up_model(self.model_anchor, self.arch_name, self.dataset_name, split2=False)

        self.transform = get_tta_transforms(self.dataset_name) 
        self.mt = cfg.M_TEACHER.MOMENTUM
        self.alpha = 0.2
        self.rst = 0.01
        self.ap = 0.92

        self.output_features_old = None

        self.model = nn.DataParallel(self.model)

        self.CP_methods = cfg.CP_METHODS
        self.CP_alhpa = cfg.CP_ALPHA

        self.cal_x, self.cal_y = init_cal_from_source_test(self.dataset_name, cfg.DATA_DIR)

        self.stu_CP = SmoothConformalPredictor(score_function=THR(), model=self.model, cal_x=self.cal_x, cal_y=self.cal_y, temperature=1)
        self.tea_CP = SmoothConformalPredictor(score_function=THR(), model=self.model_ema, cal_x=self.cal_x, cal_y=self.cal_y, temperature=1)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        # Use this line to also restore the teacher model                         
        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        
        self.stu_CP = SmoothConformalPredictor(score_function=THR(), model=self.model, cal_x=self.cal_x, cal_y=self.cal_y, temperature=1)
        self.tea_CP = SmoothConformalPredictor(score_function=THR(), model=self.model_ema, cal_x=self.cal_x, cal_y=self.cal_y, temperature=1)


    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        
        labels = x[1]
        x = x[0]

        outputs = self.model(x)
        outputs_ema = self.model_ema(x)
        imgs_test_aug = self.transform(x)
        outputs_aug = self.model(imgs_test_aug)
        
        
        loss_certain = torch.tensor(0.0).to(self.device)
        loss_cal = torch.tensor(0.0).to(self.device)
        self.stu_CP.smooth_calibrate(alpha=self.CP_alhpa)
        stu_pred_set, stu_smooth_predict = self.stu_CP.conformal_predict(outputs)
        if self.CP_methods == "THR":
            self.stu_CP.smooth_calibrate(alpha=self.CP_alhpa)
            stu_pred_set, stu_smooth_predict = self.stu_CP.conformal_predict(outputs)
        elif self.CP_methods == "NexCP":
            self.stu_CP.smooth_calibrate(alpha=self.CP_alhpa)
            stu_pred_set, stu_smooth_predict = self.stu_CP.conformal_predict(outputs)
            adapt_indices = self.updae_cal_set(x, outputs, stu_pred_set, up_num=10)
        elif self.CP_methods == "QTC":
            # Step1: QTC-Target
            self.CP_alhpa = torch.tensor(self.CP_alhpa)
            target_max_pred = outputs.max(1)[0]
            n = target_max_pred.shape[0]
            q_level = torch.ceil((n+1)*(self.CP_alhpa))/n
            k = int(q_level * n)
            target_sorted_array = target_max_pred.sort()[0]
            q_DQ = target_sorted_array[k]
            source_max_pred = self.model(self.cal_x).detach().max(1)[0]
            beta = (source_max_pred >= q_DQ).float().sum() / source_max_pred.shape[0]
            n = source_max_pred.shape[0]
            threshold = torch.ceil((n+1)*(1 - beta))/n
            k = int(threshold * n)
            source_pred = self.model(self.cal_x).detach()[torch.arange(self.cal_x.shape[0]), self.cal_y]
            source_sorted_array = source_pred.sort()[0]
            QTC_T = source_sorted_array[k]
            # Step2: QTC-Source
            q_level = torch.ceil((n+1)*(1 - self.CP_alhpa))/n
            k = int(q_level * n)
            q_DP = source_sorted_array[k]
            beta = (target_max_pred <= q_DP).float().sum() / target_max_pred.shape[0]
            n = source_max_pred.shape[0]
            threshold = torch.ceil((n+1)*(1 - beta))/n
            k = int(threshold * n)
            QTC_S = source_sorted_array[k]
            # Step3: max
            q_hat = torch.max(QTC_T, QTC_S)
            print(f"QTC-Target: {QTC_T}, QTC-Source: {QTC_S}, QTC: {q_hat}")
            # Step4: QTC-predict
            stu_pred_set = (outputs >= q_hat)
        elif self.CP_methods == "TUI":
            output_feature = self.feature_extractor(x).detach().mean(0)
            output_feature_src = self.feature_extractor(self.cal_x).detach().mean(0)
            output_feature_anchor = self.feature_extractor_src(x).detach().mean(0)
            output_feature_src_anchor = self.feature_extractor_src(self.cal_x).detach().mean(0)
            calib_features = torch.cat([output_feature_src, output_feature_src_anchor], dim=0).softmax(0)
            target_features = torch.cat([output_feature, output_feature_anchor], dim=0).softmax(0)
            m = 0.5 * (calib_features + target_features)
            js = 0.5 * (F.kl_div(calib_features.log(), m) + F.kl_div(target_features.log(), m))
            diff = self.lamda3 * js
            self.stu_CP.smooth_calibrate(self.CP_alhpa)
            q_hat = self.stu_CP.q_hat - diff
            stu_pred_set = (outputs >= q_hat)
            
            
        weights = stu_pred_set.sum(1) * (1.0 / (1.0 - stu_pred_set.sum(1).max())) + stu_pred_set.sum(1).max()/(stu_pred_set.sum(1).max()-1)
        if stu_pred_set.sum(1).max() == 1:
            weights = stu_pred_set.sum(1)
        certain_indices = stu_pred_set.sum(1) >= 1

        # loss: 
        loss_certain = (weights[certain_indices] * self.softmax_entropy(outputs[certain_indices])).mean(0)
        output_pred = outputs.max(1)[1]
        output_features = self.feature_extractor(x)
        # loss_certain_prototype = F.mse_loss(output_features, self.prototypes_src[output_pred].squeeze(1))
        loss_certain_prototype = (weights[certain_indices] * (output_features[certain_indices] - self.prototypes_src[output_pred[certain_indices]].squeeze(1)).pow(2).sum(1)).mean(0)
        loss_sce = (0.5 * symmetric_cross_entropy(outputs_aug, outputs_ema)).mean(0)

        
        output_src = self.model(self.cal_x)
        source_replay_loss = F.cross_entropy(output_src, self.cal_y)


        # Student update
        loss = 0
        if self.CP_methods == "None":
            loss = loss_sce
        else:
            loss = loss_sce + self.lamda1 * loss_certain_prototype + self.lamda2 * loss_certain
        # loss = loss_sce  +  100 * source_replay_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Teacher update, moving average
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)

        beta = 0.8
        final_output = beta * outputs + (1 - beta) * outputs_aug


        losses = {
            'sce_loss': loss_sce,
            'loss_certain_prototype': loss_certain_prototype,
            'loss': loss,
            # 'certain_indices': certain_indices.sum(),
        }


        return final_output, losses, stu_pred_set


    def updae_cal_set(self, x, logits, pred_set, up_num):
        assert up_num <= x.shape[0]
        ############### Select example ####################
        selected_indices = torch.randperm(x.shape[0])[:up_num]
        self.cal_x = torch.cat([self.cal_x[up_num:], x[selected_indices]], dim=0)
        self.cal_y = torch.cat([self.cal_y[up_num:], logits.max(1)[1][selected_indices]], dim=0)
        #####################################
        unselected_indices = torch.ones(x.shape[0], dtype=torch.bool)
        unselected_indices[selected_indices] = False
        return unselected_indices
    
    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)
        # enable all trainable
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            else:
                m.requires_grad_(True)
        return self.model
    


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"




class SmoothConformalPredictor(SplitPredictor):
    def __init__(self, score_function, cal_x, cal_y, model=None, temperature=1):
        super().__init__(score_function, model, temperature)
        self.temperature = temperature
        self.cal_x = cal_x
        self.cal_y = cal_y
        self.dispersion = 0.1 
        self.rho = 0.99

    def conformal_predict(self, logits): 
        mask = logits >= (self.q_hat)
        smooth_predict =  torch.sigmoid((logits-self.q_hat)/self.temperature)
        return mask, smooth_predict

    
    def smooth_calibrate(self, alpha):
        self.alpha = torch.tensor(alpha)
        self._model.eval()
        with torch.no_grad():
            logits = self._model(self.cal_x).detach()
        self.smooth_calculate_threshold(logits, self.cal_y)
        return logits
    
    def smooth_calculate_threshold(self, logits, labels):
        logits = logits.to(self._device)
        labels = labels.to(self._device)
        scores = self.score_function(logits, labels)
        self.q_hat = self.calculate_conformal_value(scores)



    def calculate_conformal_value(self, scores):
        n = scores.shape[0]
        q_level = np.ceil((n+1)*(self.alpha))/n
        k = int(q_level * n)
        sorted_array = scores.sort()[0]
        return sorted_array[k]



def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


@torch.jit.script
def symmetric_cross_entropy(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    """Mario Dobler et al. RMT CVPR-23"""
    alpha = 0.0
    loss = - (1 - alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) 
    loss += - alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)
    return loss


def collect_params(model):
    """Collect all trainable parameters.

    Walk the model's modules and collect all parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if True:#isinstance(m, nn.BatchNorm2d): collect all 
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)



def init_cal_from_source_test(dataset='cifar10',data_root_dir='./data'):


    selected_indices = []
    if dataset in ["cifar10", "cifar10_c"]:
        test_data = datasets.CIFAR10(root=data_root_dir,
                                                      train=True,
                                                      download=True,
                                                      transform=transforms.ToTensor())
        for class_label in range(10):
            class_indices = torch.where(torch.tensor(test_data.targets) == class_label)[0]
            selected_indices.extend(class_indices[torch.randperm(len(class_indices))[:5]])
    elif dataset in ["cifar100", "cifar100_c"]:
        test_data = datasets.CIFAR100(root=data_root_dir,
                                                       train=True,
                                                       download=True,
                                                       transform=transforms.ToTensor())
        for class_label in range(100):
            class_indices = torch.where(torch.tensor(test_data.targets) == class_label)[0]
            selected_indices.extend(class_indices[torch.randperm(len(class_indices))[:3]])
    elif dataset in ["imagenet", "imagenet_c", "imagenet_k", "ccc"]:
        test_data = datasets.ImageNet(root="/data3/share/imagenet/imagenet/imagenet",
                                                    #    train=False,
                                                       transform=transforms.Compose([
                                                              transforms.Resize(256),
                                                              transforms.CenterCrop(224),
                                                              transforms.ToTensor()
                                                         ]))
        # for class_label in range(1000):
            # class_indices = torch.where(torch.tensor(test_data.targets) == class_label)[0]
            # selected_indices.extend(class_indices[torch.randperm(len(class_indices))[:1]])
        selected_indices.extend(torch.randperm(len(test_data))[:100])
    
    
    subset_test_data = torch.utils.data.Subset(test_data, selected_indices)
    test_loader = torch.utils.data.DataLoader(subset_test_data, batch_size=len(subset_test_data), shuffle=False)
    images, labels = next(iter(test_loader))
    indices = list(range(len(images)))
    random.shuffle(indices)

    images_shuffled = images[indices]
    labels_shuffled = labels[indices]

    print(f"Calibration set size: {len(images_shuffled)}")
    
    return images_shuffled.cuda(), labels_shuffled.cuda()



class THR(BaseScore):
    """
    Threshold conformal predictors (Sadinle et al., 2016).
    paper : https://arxiv.org/abs/1609.00451.
    
    :param score_type: a transformation on logits. Default: "softmax". Optional: "softmax", "Identity", "log_softmax" or "log".
    """

    def __init__(self, score_type="Identity"):
        
        super().__init__()
        self.score_type = score_type
        if score_type == "Identity":
            self.transform = lambda x: x
        elif score_type == "softmax":
            self.transform = lambda x: torch.softmax(x, dim=- 1)
        elif score_type == "log_softmax":
            self.transform = lambda x: torch.log_softmax(x, dim=-1)
        elif score_type == "log":
            self.transform = lambda x: torch.log(x)
        else:
            raise NotImplementedError

    def __call__(self, logits, label=None):
        assert len(logits.shape) <= 2, "dimension of logits are at most 2."
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)
        temp_values = self.transform(logits)
        if label is None:
            return self.__calculate_all_label(temp_values)
        else:
            return self.__calculate_single_label(temp_values, label)

    def __calculate_single_label(self, temp_values, label):
        return temp_values[torch.arange(temp_values.shape[0], device=temp_values.device), label]

    def __calculate_all_label(self, temp_values):
        return temp_values