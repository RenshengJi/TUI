import os
import torch
import logging
import numpy as np
import methods

from models.model import get_model
from utils.eval_utils import get_accuracy, eval_domain_dict
from utils.registry import ADAPTATION_REGISTRY
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes, ckpt_path_to_domain_seq

logger = logging.getLogger(__name__)

def evaluate(description):


    load_cfg_from_args(description)
    valid_settings = ["reset_each_shift",           # reset the model state after the adaptation to a domain
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                      "mixed_domains",              # consecutive test samples are likely to originate from different domains
                      "correlated",                 # sorted by class label
                      "mixed_domains_correlated",   # mixed domains + sorted by class label
                      "gradual_correlated",         # gradual domain shifts + sorted by class label
                      "reset_each_shift_correlated"
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # get the base model and its corresponding input pre-processing (if available)
    base_model, model_preprocess = get_model(cfg, num_classes, device)

    # append the input pre-processing to the base model
    base_model.model_preprocess = model_preprocess

    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET == "domainnet126":
        # extract the domain sequence for a specific checkpoint.
        domain_sequence = ckpt_path_to_domain_seq(ckpt_path=cfg.MODEL.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in ["imagenet_d", "imagenet_d109"] and not cfg.CORRUPTION.TYPE[0]:
        # domain_sequence = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        domain_sequence = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        domain_sequence = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {domain_sequence}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    domain_seq_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else domain_sequence

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in ["cifar10_c", "cifar100_c", "imagenet_c"] and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    errs = []
    errs_5 = []
    domain_dict = {}

    cover_stu_sets = []
    cover_tea_sets = []
    num_stu_sets = []
    num_tea_sets = []

    accss = []
    max_scores = []
    Ineffss = []
    is_accss = []
    nllss = []


    # start evaluation
    for i_dom, domain_name in enumerate(domain_seq_loop):
        # if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
        if "reset_each_shift" in cfg.SETTING:
            try:
                model.reset()
                logger.info("resetting model")
            except AttributeError:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")

        for severity in severities:
            test_data_loader = get_test_loader(setting=cfg.SETTING,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               dataset_name=cfg.CORRUPTION.DATASET,
                                               preprocess=model_preprocess,
                                               data_root_dir=cfg.DATA_DIR,
                                               domain_name=domain_name,
                                               domain_names_all=domain_sequence,
                                               severity=severity,
                                               num_examples=cfg.CORRUPTION.NUM_EX,
                                               rng_seed=cfg.RNG_SEED,
                                               delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                                               batch_size=cfg.TEST.BATCH_SIZE,
                                               shuffle=False,
                                               workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()))
            
            # evaluate the model
            acc, domain_dict, num_samples, cover_stu_set, num_stu_set, max_score, accs, Ineffs, is_accs, nlls= get_accuracy(model,
                                                         data_loader=test_data_loader,
                                                         dataset_name=cfg.CORRUPTION.DATASET,
                                                         domain_name=domain_name,
                                                         setting=cfg.SETTING,
                                                         domain_dict=domain_dict,
                                                         print_every=cfg.PRINT_EVERY,
                                                         device=device)
            
            accss.extend(accs)
            max_scores.extend(max_score)
            Ineffss.extend(Ineffs)
            is_accss.extend(is_accs)
            nllss.extend(nlls)

            err = 1. - acc
            errs.append(err)
            if severity == 5 and domain_name != "none":
                errs_5.append(err)
            
            cover_stu_sets.append(cover_stu_set)
            num_stu_sets.append(num_stu_set)

            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={num_samples}]: {err:.2%}")
            logger.info(f"Student coverage % [{domain_name}{severity}][#samples={num_samples}]: {cover_stu_set:.2%}")
            logger.info(f"Student average #samples [{domain_name}{severity}][#samples={num_samples}]: {num_stu_set:.2f}")


    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs)}, mean error at 5: {np.mean(errs_5)}, mean error student: {np.mean(cover_stu_sets)}, mean error teacher: {np.mean(cover_tea_sets)}, mean #samples student: {np.mean(num_stu_sets)}, mean #samples teacher: {np.mean(num_tea_sets)}")
    else:
        logger.info(f"mean error: {np.mean(errs)}")



    path = 'result/' + cfg.MODEL.ADAPTATION 
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path + '/errs.txt', 'a') as f:
        f.write(str(errs) + '\n')
    with open(path + '/cover_stu_sets.txt', 'a') as f:
        f.write(str(cover_stu_sets) + '\n')
    with open(path + '/num_stu_sets.txt', 'a') as f:
        f.write(str(num_stu_sets) + '\n')


    # NLL
    nll = np.mean(nllss)
    
    # Brier Score
    brier_score = np.mean(np.square(np.array(is_accss) - np.array(max_scores)))

    # ECE(Expected Calibration Error)
    max_scores = np.array(max_scores)
    is_accss = np.array(is_accss)
    bins = np.linspace(0, 1, 11)
    accs = []
    mean_scores = []
    for i in range(10):
        accs.append(np.mean(is_accss[(max_scores >= bins[i]) & (max_scores < bins[i+1])]))
        mean_scores.append(np.mean(max_scores[(max_scores >= bins[i]) & (max_scores < bins[i+1])]))
    # nan
    accs = np.nan_to_num(accs)
    mean_scores = np.nan_to_num(mean_scores)
    ece = np.sum(np.abs(np.array(accs) - np.array(mean_scores))) / len(accs)


    with open(path + '/result.txt', 'a') as f:
        # f.write(str(round(np.mean(errs)*100, 4) + '&' + str(round(nll, 2)) + '&' + str(round(brier_score, 2)) + '&' + str(round(ece, 2)) + '\n')
        f.write(str(round(np.mean(errs)*100, 2)) + ' & ' + str(round(np.mean(cover_stu_sets)*100, 2) ) + ' & ' + str(round(np.mean(num_stu_sets), 2) ) + ' & ' + str(round(nll, 2)) + ' & ' + str(round(brier_score, 2)) + ' & ' + str(round(ece, 2) ) + '\n')
        # f.write(str(round(np.mean(errs), 4)) + ' & ' + str(round(np.mean(cover_stu_sets), 4) ) + ' & ' + str(round(np.mean(num_stu_sets), 4) ) + ' & ' + str(round(nll, 4)) + ' & ' + str(round(brier_score, 4)) + ' & ' + str(round(ece, 4) ) + '\n')




    if "mixed_domains" in cfg.SETTING and len(domain_dict.values()) > 0:
        # print detailed results for each domain
        eval_domain_dict(domain_dict, domain_seq=domain_sequence)


if __name__ == '__main__':
    evaluate('"Evaluation.')
