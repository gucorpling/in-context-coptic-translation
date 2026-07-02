import argparse
from pathlib import Path
import itertools

import pandas as pd
if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("scores_files", nargs="+", type=str)

    args = arg_parser.parse_args()

    scores_files = args.scores_files

    # Make pairs of scores files
    # FIX: assuming one_pair
    scorefile_1, scorefile_2 = scores_files
    _,_, model_name1, setting_1, split1, *_ = scorefile_1.split("_")
    _,_, model_name2, setting_2, split2, *_ = scorefile_2.split("_")
    assert model_name1 == model_name2
    split1 = split1.split(".")[0]
    split2 = split2.split(".")[0]
    model_name = model_name1
    print(split1, split2)
    assert split1 == split2
    split = split1
    assert setting_1 != setting_2

    print(f"Comparing {setting_1} and {setting_2} with files {scorefile_1} and {scorefile_2}")
    df_1 = pd.read_csv(scorefile_1, sep="\t")
    df_2 = pd.read_csv(scorefile_2, sep="\t")

    # Make a new df with delta between the two settings
    paired_df = df_1[["source", "reference"]].copy()

    paired_df["setting_1"] = setting_1
    paired_df["score_1"] = df_1["BERTScore_F1"]
    paired_df["prediction_1"] = df_1["predictions"]


    paired_df["setting_2"] = setting_2
    paired_df["score_2"] = df_2["BERTScore_F1"]
    paired_df["prediction_2"] = df_2["predictions"]

    paired_df["delta"] = paired_df["score_2"] - paired_df["score_1"]

    print(paired_df.head())

    paired_df.to_csv(f"Dev_Paired/{setting_1}_{setting_2}_{model_name}_{split}.tsv", sep="\t", index=False)




