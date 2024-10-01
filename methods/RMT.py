from copy import deepcopy
import logging

import torch
import torch.nn as nn
import torch.jit

import random
import tqdm
import PIL
import torchvision.transforms as transforms
import methods.my_transforms as my_transforms
from time import time
from utils.losses import Entropy

import logging
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)
import torchvision.datasets as datasets

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from augmentations.transforms_cotta import get_tta_transforms
from utils.losses import SymmetricCrossEntropy
from datasets.data_loading import get_source_loader


from torchcp.classification.scores import THR
from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores.base import BaseScore
from models.model import split_up_model
import os


@ADAPTATION_REGISTRY.register()
class RMT(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.model = model
        self.softmax_entropy = Entropy()
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
        self.contrast_mode = cfg.CONTRAST.MODE
        self.temperature = cfg.CONTRAST.TEMPERATURE
        self.base_temperature = self.temperature
        self.projection_dim = cfg.CONTRAST.PROJECTION_DIM
        self.lambda_ce_src = cfg.RMT.LAMBDA_CE_SRC
        self.lambda_ce_trg = cfg.RMT.LAMBDA_CE_TRG
        self.lambda_cont = cfg.RMT.LAMBDA_CONT
        self.m_teacher_momentum = cfg.M_TEACHER.MOMENTUM
        # arguments neeeded for warm up
        self.warmup_steps = cfg.RMT.NUM_SAMPLES_WARM_UP // batch_size_src
        self.final_lr = cfg.OPTIM.LR
        arch_name = cfg.MODEL.ARCH
        ckpt_path = cfg.MODEL.CKPT_PATH
        self.entropy = Entropy()

        self.tta_transform = get_tta_transforms(self.dataset_name)

        # setup loss functions
        self.symmetric_cross_entropy = SymmetricCrossEntropy(0.5)

        # Setup EMA model
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)

        # split up the model
        self.feature_extractor, self.classifier = split_up_model(self.model, arch_name, self.dataset_name)
        self.feature_extractor_src, self.classifier_src = split_up_model(self.model_anchor, arch_name, self.dataset_name, split2=False)


        # define the prototype paths
        proto_dir_path = os.path.join(cfg.CKPT_DIR, "prototypes")
        if self.dataset_name == "domainnet126":
            fname = f"protos_{self.dataset_name}_{ckpt_path.split(os.sep)[-1].split('_')[1]}.pth"
        else:
            fname = f"protos_{self.dataset_name}_{arch_name}.pth"
        fname = os.path.join(proto_dir_path, fname)

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

        # setup projector
        if self.dataset_name == "domainnet126":
            # do not use a projector since the network already clusters the features and reduces the dimensions
            self.projector = nn.Identity()
        else:
            num_channels = self.prototypes_src.shape[-1]
            self.projector = nn.Sequential(nn.Linear(num_channels, self.projection_dim), nn.ReLU(),
                                           nn.Linear(self.projection_dim, self.projection_dim)).to(self.device)
            self.optimizer.add_param_group({'params': self.projector.parameters(), 'lr': self.optimizer.param_groups[0]["lr"]})
        

        self.mt = cfg.M_TEACHER.MOMENTUM
        self.alpha = cfg.alpha
        self.cal_x, self.cal_y = init_cal_from_source_test(self.dataset_name, cfg.DATA_DIR)

        self.CP_methods = cfg.CP_METHODS
        self.CP_alhpa = cfg.CP_ALPHA
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

        imgs_test = x

        self.optimizer.zero_grad()

        # forward original test data
        features_test = self.feature_extractor(imgs_test)
        outputs_test = self.classifier(features_test)

        # forward augmented test data
        features_aug_test = self.feature_extractor(self.tta_transform((imgs_test)))
        outputs_aug_test = self.classifier(features_aug_test)

        # forward original test data through the ema model
        outputs_ema = self.model_ema(imgs_test)

        with torch.no_grad():
            # dist[:, i] contains the distance from every source sample to one test sample
            dist = F.cosine_similarity(
                x1=self.prototypes_src.repeat(1, features_test.shape[0], 1),
                x2=features_test.view(1, features_test.shape[0], features_test.shape[1]).repeat(self.prototypes_src.shape[0], 1, 1),
                dim=-1)

            # for every test feature, get the nearest source prototype and derive the label
            _, indices = dist.topk(1, largest=True, dim=0)
            indices = indices.squeeze(0)

        features = torch.cat([self.prototypes_src[indices],
                              features_test.view(features_test.shape[0], 1, features_test.shape[1]),
                              features_aug_test.view(features_test.shape[0], 1, features_test.shape[1])], dim=1)
        loss_contrastive = self.contrastive_loss(features=features, labels=None)

        loss_self_training = (0.5 * self.symmetric_cross_entropy(outputs_test, outputs_ema) + 0.5 * self.symmetric_cross_entropy(outputs_aug_test, outputs_ema)).mean(0)


        loss_certain = torch.tensor(0.0).to(self.device)
        loss_cal = torch.tensor(0.0).to(self.device)
        self.stu_CP.smooth_calibrate(alpha=self.CP_alhpa)
        stu_pred_set, stu_smooth_predict = self.stu_CP.conformal_predict(outputs_test)
        if self.CP_methods == "THR":
            self.stu_CP.smooth_calibrate(alpha=self.CP_alhpa)
            stu_pred_set, stu_smooth_predict = self.stu_CP.conformal_predict(outputs_test)
        elif self.CP_methods == "NexCP":
            self.stu_CP.smooth_calibrate(alpha=self.CP_alhpa)
            stu_pred_set, stu_smooth_predict = self.stu_CP.conformal_predict(outputs_test)
            adapt_indices = self.updae_cal_set(x, outputs_test, stu_pred_set, up_num=10)
        elif self.CP_methods == "QTC":
            # Step1: QTC-Target
            self.CP_alhpa = torch.tensor(self.CP_alhpa)
            target_max_pred = outputs_test.max(1)[0]
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
            # Step3: min
            q_hat = torch.min(QTC_T, QTC_S)
            # Step4: QTC-predict
            stu_pred_set = (outputs_test >= q_hat)
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
            stu_pred_set = (outputs_test >= q_hat)
            
            
        weights = stu_pred_set.sum(1) * (1.0 / (1.0 - stu_pred_set.sum(1).max())) + stu_pred_set.sum(1).max()/(stu_pred_set.sum(1).max()-1)
        if stu_pred_set.sum(1).max() == 1:
            weights = stu_pred_set.sum(1)
        certain_indices = stu_pred_set.sum(1) >= 1


        # loss: 
        loss_certain = (weights[certain_indices] * self.entropy((outputs_test[certain_indices]))).mean(0)
        output_pred = outputs_test.max(1)[1]
        output_features = self.feature_extractor(x)
        # loss_certain_prototype = F.mse_loss(output_features, self.prototypes_src[output_pred].squeeze(1))
        loss_certain_prototype = (weights[certain_indices] * (output_features[certain_indices] - self.prototypes_src[output_pred[certain_indices]].squeeze(1)).pow(2).sum(1)).mean(0)


        # Student update
        loss_trg = 0
        if self.CP_methods == "None":
            loss_trg = self.lambda_ce_trg * loss_self_training + self.lambda_cont * loss_contrastive
        else:
            loss_trg = self.lambda_ce_trg * loss_self_training + self.lambda_cont * loss_contrastive + self.lamda1 * loss_certain_prototype + self.lamda2 * loss_certain
        loss_trg.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Teacher update, moving average
        self.model_ema = update_ema_variables(ema_model = self.model_ema, model = self.model, alpha_teacher=self.mt)


        losses = {
            'loss': loss_trg
        }

        if self.CP_methods == "None":
            final_outputs = outputs_test + outputs_ema
        else:
            final_outputs = 0.2 * outputs_test + 0.8 * outputs_ema

        return final_outputs, losses, stu_pred_set


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
            else:
                m.requires_grad_(True)
        return self.model
    

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def warmup(self):
        logger.info(f"Starting warm up...")
        for i in range(self.warmup_steps):
            #linearly increase the learning rate
            for par in self.optimizer.param_groups:
                par["lr"] = self.final_lr * (i+1) / self.warmup_steps

            # sample source batch
            try:
                batch = next(self.src_loader_iter)
            except StopIteration:
                self.src_loader_iter = iter(self.src_loader)
                batch = next(self.src_loader_iter)

            imgs_src, labels_src = batch[0], batch[1]
            imgs_src, labels_src = imgs_src.to(self.device), labels_src.to(self.device).long()

            # forward the test data and optimize the model
            outputs = self.model(imgs_src)
            outputs_ema = self.model_ema(imgs_src)
            loss = self.symmetric_cross_entropy(outputs, outputs_ema).mean(0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.m_teacher_momentum)

        logger.info(f"Finished warm up...")
        for par in self.optimizer.param_groups:
            par["lr"] = self.final_lr


    # Integrated from: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def contrastive_loss(self, features, labels=None, mask=None):
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        contrast_feature = self.projector(contrast_feature)
        contrast_feature = F.normalize(contrast_feature, p=2, dim=1)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    


@torch.jit.script
def softmax_entropy_cifar(x, x_ema) -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def softmax_entropy_imagenet(x, x_ema) -> torch.Tensor:
    return -0.5*(x_ema.softmax(1) * x.log_softmax(1)).sum(1)-0.5*(x.softmax(1) * x_ema.log_softmax(1)).sum(1) 


@torch.jit.script
def softmax_cross_entropy(x, x_ema):# -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


@torch.jit.script
def softmax_shannon_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1) # Shannon entrophy


@torch.jit.script
def symmetric_cross_entropy(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    """Mario Dobler et al. RMT CVPR-23"""
    alpha = 0.5
    loss = - (1 - alpha) * (x_ema.softmax(1) * x.log_softmax(1)).sum(1) 
    loss += - alpha * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)
    return loss

@torch.jit.script
def symmetric_cross_entropy2(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    """Mario Dobler et al. RMT CVPR-23"""
    alpha = 0.5
    loss = - (1 - alpha) * (x_ema * torch.log(x)).sum(1) 
    loss += - alpha * (x * torch.log(x_ema)).sum(1)
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
            selected_indices.extend(class_indices[torch.randperm(len(class_indices))[:1]])
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
