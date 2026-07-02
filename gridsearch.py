from sklearn import svm, datasets
from sklearn.model_selection import ParameterGrid
import os
import pandas as pd

from tqdm import tqdm
from openai import OpenAI

import lrmtdata
import util
from metrics import Metrics
from gpt import infer_and_extract_output

ud_param_grid = [
  {'collapse': [True, False], 
   'conllu': [None, 'only', 'first', 'last'], 
   'duplicate': ['ordinal', 'subscript'],
   'upos': ['content', 'plus', 'other']},
 ]

lex_param_grid = [
    {'entry_langs': ['en', 'en,de', 'en,fr', 'en,de,fr'],
     'max_entries': [50, 100],
     'max_senses_per_entry': [10, 20],
     'dedup_senses': [True, False]},
]

results = []
model = "gpt-4.1-mini-2025-04-14"
lexicon_file = "data/lexicon.xml"
cop_file = "data/ud_coptic/top_ten_cop.txt"
eng_file = "data/ud_coptic/top_ten_eng.txt"
conllu_file = "data/ud_coptic/top_ten.conllu"
config_path = "configs/best.components.ini"
config = util.get_components_config(config_path)

def run_one_configuration(grid_config):
    prompt_dataset = lrmtdata.get_prompt_dataset(cop_file, eng_file, conllu_file, lexicon_file, config, grid_config)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=openai_api_key)
    i = 0
    outputs = list()
    for prompt in tqdm(prompt_dataset["prompt"]):
        output = infer_and_extract_output(client, model, prompt)
        outputs.append(output)
        i = i + 1
        if (i % 10) == 0:
            print(f"Completed {i} of {len(prompt_dataset['prompt'])} prompts")
            translation_checkpoint = prompt_dataset.select(range(0, i))
            translation_checkpoint = translation_checkpoint.add_column("predictions", [o["text"] for o in outputs])
            #translation_checkpoint.to_csv(output_file, sep="\t", index=False)

    translation_dataset = prompt_dataset.add_column("predictions", [o["text"] for o in outputs])
    # translation_dataset.to_csv(output_file, sep="\t", index=False)
    # Evaluate same as evaluate-gpt.py

    return translation_dataset
    

for grid_config in ParameterGrid(lex_param_grid):
    translations = run_one_configuration(grid_config)
    # evaluate_all_metrics returns a summary dict with 4 dif scores
    score = Metrics.evaluate_all_metrics(translations["predictions"], translations["reference"])

    results.append({
        **grid_config,
        **score
    })

df = pd.DataFrame(results)
df.to_csv("lex_grid_search_results_top_10.csv", index=False)

