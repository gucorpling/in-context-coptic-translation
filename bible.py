import json
from pathlib import Path
import argparse

import pandas as pd
import pyconll

import util
from metrics import Metrics

if __name__ == "__main__":

    gold_dev_conllu_file = util.get_data_files('dev_gold')[2]

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dev_translations", "-dt", required=True)


    translations_file = arg_parser.parse_args().dev_translations

    results_df = pd.read_csv(translations_file, sep="\t")

    gold_conll_sentences = pyconll.load_from_file(gold_dev_conllu_file)
    parallel_ids = [sentence._meta.get('parallel_id', "") for sentence in gold_conll_sentences]

    results_df['parallel_id'] = parallel_ids
    print(results_df["parallel_id"].value_counts())
    results_df["from_bible"] = results_df["parallel_id"].apply(lambda x: x.startswith("bible"))
    print(results_df["from_bible"].value_counts())

    from_bible_df = results_df[results_df["from_bible"] == True]
    not_bible_df = results_df[results_df["from_bible"] == False]

    from_bible_file = Path(translations_file).parent.__str__() + "/bible_" + Path(translations_file).name
    not_bible_file = Path(translations_file).parent.__str__() + "/not_bible_" + Path(translations_file).name

    print(f"Writing {len(from_bible_df)} bible samples to {from_bible_file}")
    print(f"Writing {len(not_bible_df)} non-bible samples to {not_bible_file}")

    from_bible_df.to_csv(from_bible_file, sep="\t", index=False)
    not_bible_df.to_csv(not_bible_file, sep="\t", index=False)

    # evaluate separately

    bible_results = Metrics.evaluate_all_metrics(from_bible_df["predictions"].values, from_bible_df["reference"].values)
    bible_results["source"] = "bible"

    bible_json_file = from_bible_file.replace(".tsv", "_metrics.json")

    not_bible_results = Metrics.evaluate_all_metrics(not_bible_df["predictions"].values, not_bible_df["reference"].values)
    not_bible_results["source"] = "not_bible"

    not_bible_json_file = not_bible_file.replace(".tsv", "_metrics.json")

    print(f"Saving bible metrics to {bible_json_file}")
    with open(bible_json_file, "w") as f:
        json.dump(bible_results, f, indent=4)

    print(f"Saving non-bible metrics to {not_bible_json_file}")
    with open(not_bible_json_file, "w") as f:
        json.dump(not_bible_results, f, indent=4)




