"""
Builds upon: https://github.com/qinenergy/cotta
Corresponding paper: https://arxiv.org/abs/2203.13591
"""

from copy import deepcopy
import numpy as np
import torch
import random
import torch.nn as nn
import torch.jit
import torch.nn.functional as F
import os
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
import torchvision.datasets as datasets
import torch
import os
import tqdm
import logging
import torchvision.transforms as transforms
from augmentations.transforms_cotta import get_tta_transforms
from datasets.data_loading import get_source_loader
from utils.losses import SymmetricCrossEntropy
from models.model import split_up_model
from torchcp.classification.scores import THR
from torchcp.classification.predictors import SplitPredictor
from torchcp.classification.scores.base import BaseScore

logger = logging.getLogger(__name__)


def update_ema_variables(ema_model, model, alpha_teacher):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]
    return ema_model


@ADAPTATION_REGISTRY.register()
class CCoTTA(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        self.softmax_entropy = Entropy()
        self.arch_name = cfg.MODEL.ARCH
        self.CKPT_DIR = cfg.CKPT_DIR
        self.cav_num = cfg.cav_num
        self.cav_alpha = cfg.cav_alpha
        self.cav_beta = cfg.cav_beta
        self.feature_extractor, self.classifier = split_up_model(self.model, self.arch_name, self.dataset_name, split2=False)
        
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
        self.mt = cfg.M_TEACHER.MOMENTUM
        arch_name = cfg.MODEL.ARCH
        self.entropy = Entropy()
        self.symmetric_cross_entropy = SymmetricCrossEntropy(alpha=0.5)
        # EMA teacher
        self.model_ema = self.copy_model(self.model)
        for param in self.model_ema.parameters():
            param.detach_()

        self.model_state, self.optimizer_state, self.model_ema, self.model_anchor = \
            copy_model_and_optimizer(self.model, self.optimizer)
        self.feature_extractor_src, self.classifier_src = split_up_model(self.model_anchor, self.arch_name, self.dataset_name, split2=False)



        # get class-wise source domain prototypes
        proto_dir_path = os.path.join(cfg.CKPT_DIR, "prototypes")
        scav_dir_path = os.path.join(cfg.CKPT_DIR, "scavs")
        fname_proto = f"protos_{self.dataset_name}_{arch_name}.pth"
        fname_proto = os.path.join(proto_dir_path, fname_proto)
        if os.path.exists(fname_proto.replace(".pth", "_domain.pth")):
            logger.info("Loading class-wise source prototypes ...")
            self.prototypes_src_domain = torch.load(fname_proto.replace(".pth", "_domain.pth"))
            self.prototypes_src_class = torch.load(fname_proto.replace(".pth", "_class.pth"))
        else:
            os.makedirs(proto_dir_path, exist_ok=True)
            os.makedirs(scav_dir_path, exist_ok=True)
            features_src_domain = torch.tensor([])
            features_src_class = torch.tensor([])
            labels_src = torch.tensor([])
            logger.info("Extracting source prototypes ...")
            max_length = 32000 if "imagenet" in self.dataset_name else 100000
            with torch.no_grad():
                for data in tqdm.tqdm(self.src_loader):
                    x, y = data[0], data[1]
                    tmp_features = self.encoder1(x.to(self.device))
                    tmp_features_domain = tmp_features
                    tmp_features = self.encoder2(tmp_features)
                    tmp_features_class = tmp_features
                    features_src_domain = torch.cat([features_src_domain, tmp_features_domain.view(tmp_features_domain.shape[0],-1).cpu()], dim=0)
                    features_src_class = torch.cat([features_src_class, tmp_features_class.view(tmp_features_class.shape[0],-1).cpu()], dim=0)
                    labels_src = torch.cat([labels_src, y], dim=0)
                    if len(features_src_domain) > max_length:
                        break

            # create class-wise source prototypes
            self.prototypes_src_domain = torch.tensor([])
            self.prototypes_src_class = torch.tensor([])
            for i in range(self.num_classes):
                mask = labels_src == i
                self.prototypes_src_domain = torch.cat([self.prototypes_src_domain, features_src_domain[mask].mean(dim=0, keepdim=True)], dim=0)
                self.prototypes_src_class = torch.cat([self.prototypes_src_class, features_src_class[mask].mean(dim=0, keepdim=True)], dim=0)
            torch.save(self.prototypes_src_domain, fname_proto.replace(".pth", "_domain.pth"))
            torch.save(self.prototypes_src_class, fname_proto.replace(".pth", "_class.pth"))


        # get relative direction of source domain categories 
        scav_dir_path = os.path.join(self.CKPT_DIR, "scavs")
        fname_scav = f"scav_inter_{self.dataset_name}_{self.arch_name}.pth"
        fname_scav = os.path.join(scav_dir_path, fname_scav)
        if os.path.exists(fname_scav):
            logger.info("Loading class-wise scavs for cav...")
            self.scav_src = torch.load(fname_scav)
        else:
            logger.info("Build source scavs ...")
            self.scav_src = torch.zeros(self.num_classes, self.num_classes, self.prototypes_src_class.size(1))
            for i in tqdm.tqdm(range(self.num_classes)):
                for j in range(self.num_classes):
                    if i != j:
                        self.scav_src[i][j] = self.prototypes_src_class[j] - self.prototypes_src_class[i]
                    else:
                        self.scav_src[i][j] = torch.zeros(1, self.prototypes_src_class.size(1))
            torch.save(self.scav_src, fname_scav)
        

        # self.model = nn.DataParallel(self.model)
        self.models = [self.model, self.model_ema]
        self.model_states, self.optimizer_state = self.copy_model_and_optimizer()
        self.transform = get_tta_transforms(self.dataset_name)

        self.cal_x, self.cal_y = init_cal_from_source_test(self.dataset_name, cfg.DATA_DIR)

        self.CP_methods = cfg.CP_METHODS
        self.CP_alhpa = cfg.CP_ALPHA
        self.stu_CP = SmoothConformalPredictor(score_function=THR(), model=self.model, cal_x=self.cal_x, cal_y=self.cal_y, temperature=1)
        self.tea_CP = SmoothConformalPredictor(score_function=THR(), model=self.model_ema, cal_x=self.cal_x, cal_y=self.cal_y, temperature=1)



    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):

        labels = x[1]
        x = x[0]

        imgs_test = x

        # sce loss
        imgs_test_aug = self.transform(imgs_test)
        outputs_aug = self.model(imgs_test_aug)
        outputs_ema = self.model_ema(imgs_test)


        # with torch.no_grad():
        if True:
            imgs_test = x
            # target sample -> student model ==> prediction 
            imgs_test = self.encoder1(imgs_test)
            features_domain = imgs_test
            features_domain_grad = imgs_test
            imgs_test = self.encoder2(imgs_test)
            features_class = imgs_test
            outputs = self.classifier(imgs_test)


        # Reliable samples selection
        entropy = (- outputs.softmax(1) * outputs.log_softmax(1)).sum(1)
        mask_entropy = entropy < (0.4 * np.log(self.num_classes))
        features_class = features_class[mask_entropy]
        features_domain = features_domain.view(features_domain.size(0), -1)[mask_entropy]
        outputs_ = outputs[mask_entropy]


        # get class-wise target domain prototypes
        pseudo_label = outputs_.argmax(1).cpu()
        classes = torch.unique(pseudo_label)
        classes_features_class = torch.zeros(classes.size(0), features_class.size(1)).to(self.device)
        classes_features_domain = torch.zeros(classes.size(0), features_domain.size(1))
        for i, c in enumerate(classes):
            mask_class = pseudo_label == c
            classes_features_class[i] = features_class[mask_class].mean(0)
            classes_features_domain[i] = features_domain[mask_class].mean(0)


        # Obtain the specific category of offset direction and apply constraints.
        prototypes_src_class = self.prototypes_src_class[classes].to(self.device)
        shift_direction = classes_features_class - prototypes_src_class
        shift_direction = F.normalize(shift_direction, p=2, dim=1)
        scav =  - prototypes_src_class.unsqueeze(1) + self.prototypes_src_class.unsqueeze(0).to(self.device)
        scav = F.normalize(scav, p=2, dim=2)
        loss_shifted_direction = torch.einsum("bd,bcd->bc", shift_direction, scav).mean(0).mean(0)


        # Obtain the offset direction of the overall domain and apply constraints.
        grad_outputs = torch.zeros_like(outputs)
        outputs_pred = outputs.argmax(dim=1)
        grad_outputs[range(outputs.shape[0]), outputs_pred] = 1
        grads = torch.autograd.grad(outputs, features_domain_grad, grad_outputs, create_graph=True)[0]
        grads = grads.view(grads.size(0), -1)
        features_domain_grad = features_domain_grad.view(features_domain_grad.size(0), -1)
        prototypes_src_domain = self.prototypes_src_domain
        if "imagenet" in self.dataset_name:
                prototypes_src_domain = prototypes_src_domain[:64]
        prototypes_src_domain = prototypes_src_domain.to(self.device)
        prototypes_src_domain_ = prototypes_src_domain.mean(0, keepdim=True).squeeze(0)
        features_domain_grad_ = features_domain_grad.mean(0, keepdim=True).squeeze(0)
        scav_ = (prototypes_src_domain_ - features_domain_grad_)
        grads_domain = torch.einsum('bf, f -> b', grads, scav_).abs()
        loss_domain_shift = grads_domain.mean(0)



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
            # Step3: min
            q_hat = torch.min(QTC_T, QTC_S)
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
        loss_certain = (weights[certain_indices] * self.entropy((outputs[certain_indices]))).mean(0)
        output_pred = outputs.max(1)[1]
        # 对certain_indices使用self.prototypes_src进行 mseloss计算
        output_features = self.feature_extractor(x)
        # loss_certain_prototype = F.mse_loss(output_features, self.prototypes_src[output_pred].squeeze(1))
        loss_certain_prototype = (weights[certain_indices] * (output_features[certain_indices] - self.prototypes_src_class.to(self.device)[output_pred[certain_indices]].squeeze(1)).pow(2).sum(1)).mean(0)



        # Student update
        loss_sce = (0.5 * self.symmetric_cross_entropy(outputs_aug, outputs_ema)).mean(0)
        loss = 0
        if self.CP_methods == "None":
            loss = (loss_sce + loss_shifted_direction * self.cav_alpha + loss_domain_shift * self.cav_beta)
        else:
            loss = (loss_sce + loss_shifted_direction * self.cav_alpha + loss_domain_shift * self.cav_beta) + self.lamda1 * loss_certain_prototype + self.lamda2 * loss_certain
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Teacher update
        self.model_ema = update_ema_variables(ema_model=self.model_ema, model=self.model, alpha_teacher=self.mt)

        losses = {"loss_sce": loss_sce, "loss_class": loss_shifted_direction, "loss_domain": loss_domain_shift}

        return outputs, losses, stu_pred_set


    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        imgs_test = x[0]
        return self.model_ema(imgs_test)


    def configure_model(self):
        """Configure model."""
        # self.model.train()
        self.model.eval()  # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
        # disable grad, to (re-)enable only what we update
        self.model.requires_grad_(False)

        for encoder in self.feature_extractor:
            for m in encoder.modules():
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

        for m in self.classifier.modules():
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


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    model_anchor = deepcopy(model)
    optimizer_state = deepcopy(optimizer.state_dict())
    ema_model = deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    return model_state, optimizer_state, ema_model, model_anchor