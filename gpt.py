import argparse
import os

from pathlib import Path
import logging

import pprint as pp
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from openai import OpenAI

import lrmtdata
import util
from metrics import Metrics


# def translate_coptic_to_english(dataset, model):
#     # We should be able to use any HF Decoder Model
def infer_and_extract_output(client, model,  prompt, seed):
    def response(prompt):
        return client.responses.create(
            model=model,
            # messages=[{"role": "user", "content": prompt}],
            input=prompt,
            # No use of prompt template right now
            include=["message.output_text.logprobs"],
            # TODO FIX: adjust parameters as needed
            max_output_tokens=128,
            temperature=0,
            top_logprobs=20,
            # seed=seed, # currently not supported in openai-python response API
        )
    def output_from_response(response):
        content = response.output[0].content[0],
        return {
            "text": content[0].text,
        }
    return output_from_response(response(prompt))



def get_openai_api_key():
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    else:
        print("Using provided OpenAI API key as OPENAI_API_KEY.")

if __name__ == '__main__':
    lexicon_file = "data/lexicon.xml"

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--config", "-c", required=True)
    arg_parser.add_argument("--model", "-m", required=True)
    arg_parser.add_argument("--split", "-s", required=True)
    arg_parser.add_argument("--output", "-o", required=True)
    arg_parser.add_argument("--seed", type=int, default=42) # currently not functional

    args = arg_parser.parse_args()
    logger.info(f"Args {pp.pformat(args)}")

    # model = "gpt-4.1-mini-2025-04-14"
    # split = "dev"
    # config_path = "configs/baseline.components.ini"
    # output_dir = "results/..."
    # seed = 42

    config_path = args.config
    model = args.model
    split = args.split
    output_dir = args.output
    seed = args.seed

    cop_file, eng_file, conllu_file = util.get_data_files(split)

    config = util.get_components_config(config_path)

    openai_api_key = get_openai_api_key()

    prompt_dataset = lrmtdata.get_prompt_dataset(cop_file, eng_file, conllu_file, lexicon_file, config)

    print(prompt_dataset)
    print("prompt:", prompt_dataset["prompt"][0])
    # Use environment variable `export OPENAI_API_KEY=<key>` for the API key
    client = OpenAI(api_key=openai_api_key)
    i = 0
    outputs = list()
    for prompt in tqdm(prompt_dataset["prompt"]):
        output = infer_and_extract_output(client, model, prompt, seed)
        outputs.append(output)
        i = i + 1
        if (i % 10) == 0:
            print(f"Completed {i} of {len(prompt_dataset['prompt'])} prompts")
            translation_checkpoint = prompt_dataset.select(range(0, i))
            translation_checkpoint = translation_checkpoint.add_column("predictions", [o["text"] for o in outputs])
            util.save_experiment(output_dir, split, model, config_path, config, {}, translation_checkpoint, seed)

    translation_dataset = prompt_dataset.add_column("predictions", [o["text"] for o in outputs])
    util.save_experiment(output_dir, split, model, config_path, config, {}, translation_dataset, seed)
    # Evaluate same as evaluate-gpt.py
    results = Metrics.evaluate_all_metrics(translation_dataset["predictions"], translation_dataset["reference"])
    pp.pprint(results)
    util.save_experiment(output_dir, split, model, config_path, config, results, translation_dataset, seed)


