import configparser
import json
import random
from pathlib import Path
import logging

import numpy as np
import torch
from transformers import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    set_seed(seed)

def get_components_config(filepath):
    components_config = configparser.ConfigParser()

    components_config.read(filepath)
    config = dict()
    config.update(dict(components_config.items("lexicon")))
    if config["lexicon.enabled"] == 'True':
        config["max_entries"] = int(config["max_entries"])
        config["max_senses_per_entry"] = int(config["max_senses_per_entry"])
        config["entry_langs"] = config["entry_langs"].split(",")
        config["dedup_senses"] = config["dedup_senses"] == 'True'

    config.update(dict(components_config.items("dependency")))
    config.update(dict(components_config.items("constructions")))
    config.update(dict(components_config.items("conllu")))
    return config

def clean_string(string):
    return string.strip("\\n\n\'\"")

def decode_string(string):
    return clean_string(string.encode("utf-8").decode("unicode-escape"))

def decode_dataset(dataset):
    for column in dataset.column_names:
        dataset = dataset.map(
            lambda x: {
                column : x[column].encode("utf-8").decode("unicode-escape")
            }

        )

    return dataset

def encode_dataset(dataset):
    for column in dataset.column_names:
        dataset = dataset.map(
            lambda x: {
                column : x[column].encode("unicode-escape").decode("utf-8")
            }

        )

    return dataset

def get_data_files(split):
    match split:
        case "mini":
            cop_file = "data/ud_coptic/ud_coptic_mini.txt"
            eng_file = "data/ud_coptic/ud_coptic_mini_en.txt"
            conllu_file = "data/ud_coptic/ud_coptic_mini_pred.conllu"

        case "dev":
            cop_file = "data/ud_coptic/ud_coptic_dev.txt"
            eng_file = "data/ud_coptic/ud_coptic_dev_en.txt"
            conllu_file = "data/ud_coptic/ud_coptic_dev_pred.conllu"

        case "dev_gold":
            cop_file = "data/ud_coptic/ud_coptic_dev.txt"
            eng_file = "data/ud_coptic/ud_coptic_dev_en.txt"
            conllu_file = "data/ud_coptic/ud_coptic_dev_gold.conllu"

        case "test":
            cop_file = "data/ud_coptic/ud_coptic_test.txt"
            eng_file = "data/ud_coptic/ud_coptic_test_en.txt"
            conllu_file = "data/ud_coptic/ud_coptic_test_pred.conllu"

        case "test_gold":
            cop_file = "data/ud_coptic/ud_coptic_test.txt"
            eng_file = "data/ud_coptic/ud_coptic_test_en.txt"
            conllu_file = "data/ud_coptic/ud_coptic_test_gold.conllu"

        case "miyagawa":
            cop_file = "data/miyagawa/miyagawa_test.txt"
            eng_file = "data/miyagawa/miyagawa_test_en.txt"
            conllu_file = "data/miyagawa/miyagawa_test_pred.conllu"

        case "miyagawa_gold":
            cop_file = "data/miyagawa/miyagawa_test.txt"
            eng_file = "data/miyagawa/miyagawa_test_en.txt"
            conllu_file = "data/miyagawa/miyagawa_test_gold.conllu"

        case _:
            raise ValueError(f"Unknown split: {split}")

    return cop_file, eng_file, conllu_file


def save_experiment(output_dir, split, model_name, config_filepath, config, result, out_dataset, seed):
    logger.info(f"Saving results to {output_dir}")

    experiment = dict()
    experiment["split"] = split
    experiment["model_name"] = model_name
    #experiment["seed"] = seed # currently not functional
    experiment.update(config)
    experiment.update(result)

    config_file = Path(config_filepath).name
    config_name = config_file.split(".")[0]
    if "/" in model_name:
        model_name = model_name.split("/")[1]
    Path(output_dir).mkdir(exist_ok=True)
    experiment_file = Path(output_dir) / f"{model_name}_{config_name}_{split}.json" #_seed_{seed}.json"
    logger.info(f"Saving experiment to {experiment_file}")
    with experiment_file.open("w") as f:
        json.dump(experiment, f)

    results_file = Path(output_dir) / f"results_{model_name}_{config_name}_{split}.tsv" #_seed_{seed}.tsv"
    dataset = encode_dataset(out_dataset)
    logger.info(f"Saving results to {results_file}")
    dataset.to_csv(results_file, sep="\t", index=False)
