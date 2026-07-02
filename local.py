import argparse
import pprint as pp

from tqdm import tqdm
import logging

from transformers import pipeline
import torch

import lrmtdata
from metrics import Metrics
import util

logger = logging.getLogger(__name__)

SEED = 42
util.set_random_seed(SEED)

def translate_coptic_to_english(dataset, model, batch_size):
    # We should be able to use any HF Decoder Model
    translator = pipeline("text-generation", model=model,
                          max_new_tokens=128,
                          batch_size=batch_size,
                          # Greedy Decoding
                          do_sample=False,
                          num_beams=1)
    #translator.tokenizer.pad_token_id = str(config.eos_token_id[0])
    logger.info(f"Using pipeline {pp.pformat(translator)}")

    # TODO: use role and user for the models
    return [output for output in translator(iter(tqdm(dataset["prompt"], desc="Translating")), return_full_text=False, do_sample=False, num_beams=1)]


if __name__ == '__main__':
    # Data files
    lexicon_file = "data/lexicon.xml"

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--batch_size", "-b", type=int, required=True)
    arg_parser.add_argument("--config", "-c", required=True)
    arg_parser.add_argument("--model", "-m", required=True)
    arg_parser.add_argument("--split", "-s", required=True)
    arg_parser.add_argument("--output", "-o", required=True)

    args = arg_parser.parse_args()
    logger.info(f"Args {pp.pformat(args)}")
    batch_size = args.batch_size
    config_path = args.config
    model = args.model
    split = args.split
    output_dir = args.output

    cop_file, eng_file, conllu_file = util.get_data_files(split)

    config = util.get_components_config(config_path)
    config["batch_size"] = batch_size

    prompt_dataset = lrmtdata.get_prompt_dataset(cop_file, eng_file, conllu_file, lexicon_file, config)
    print(prompt_dataset)
    print("prompt:", prompt_dataset[0]["prompt"])

    outputs = translate_coptic_to_english(prompt_dataset, model, batch_size)
    # TODO, Fix: output is not just the translation?
    print("generation:", outputs[0][0]["generated_text"])

    translation_dataset = prompt_dataset.add_column("predictions", [o[0]["generated_text"] for o in outputs])
    # Evaluate
    evaluator = Metrics()
    results = evaluator.evaluate_all_metrics(translation_dataset["predictions"], translation_dataset["reference"])
    logger.info(pp.pformat(results))
    util.save_experiment(output_dir, split, model, config_path, config, results, translation_dataset, SEED)

