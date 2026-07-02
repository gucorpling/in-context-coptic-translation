"""
Script to rank translations at sample level, and provide metrics for analysis.
"""

import argparse
import re

import pandas as pd
from tqdm import tqdm

import util
from metrics import Metrics
from pathlib import Path

def rank_translations(results, output_file):
    samples = list()
    for source, prediction, reference in tqdm(zip(results["source"], results["predictions"], results["reference"]),
                                              desc="evaluating samples individually", total=len(results)):
        result = dict()
        # print("PREDICTION:", prediction)
        # print("REFERENCE :", reference)
        scores = Metrics().evaluate_sample(prediction, reference)
        # TODO no need of merge anymore
        for k, v in scores.items():
            result[k] = v
        samples.append(result)

    ranked_df = pd.concat([results, pd.DataFrame(samples)], axis=1)
    # FIX: does this mess up delimitation w.r.t to excel etc.?
    ranked_df["predictions"] = ranked_df["predictions"].apply(lambda x: x.encode("unicode-escape").decode("utf-8"))
    ranked_df = ranked_df.sort_values("BERTScore_F1")
    ranked_df.to_csv(output_file, sep="\t", index=False)


def extract_predictions(predictions):
    return [util.decode_string(prediction) for prediction in predictions]


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("translations", type=str)
    translations_file = arg_parser.parse_args().translations
    rank_file = Path(translations_file).parent.__str__() + "/ranked_" + Path(translations_file).name
    print(rank_file)

    results_df = pd.read_csv(translations_file, sep="\t")
    print(results_df)
    results_df.fillna("", inplace=True)
    predictions = extract_predictions(results_df["predictions"])
    rank_translations(results_df[["source", "reference", "predictions"]], rank_file)
    results = Metrics.evaluate_all_metrics(results_df["predictions"], results_df["reference"])
    print(results)

