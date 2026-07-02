import os, sys, re, codecs
from argparse import ArgumentParser
from glob import glob
from collections import defaultdict

p = ArgumentParser()
p.add_argument("-p","--partition", choices=["test","dev"], default="test", help="Partition to use")
args = p.parse_args()

folder = args.partition
folder = os.path.abspath(folder)

tsv_files = glob(os.path.join(folder,"*.tsv"))

# Cols:
# source	reference	predictions	BLEU	BERTScore_P	BERTScore_R	BERTScore_F1	from_bible	lexicon	dependency	constructions	conllu	model_name

output = []
for f in tsv_files:
    lines = open(f).read().strip().split("\n")
    for l, line in enumerate(lines):
        if l == 0: # Header
            cols = line.split("\t")
            continue
        sid = "s_" + str(l)
        parts = line.split("\t")
        bleu = parts[cols.index("BLEU")]
        bert_f1 = parts[cols.index("BERTScore_F1")]
        from_bible = parts[cols.index("from_bible")]
        lexicon = parts[cols.index("lexicon")]
        dependency = parts[cols.index("dependency")]
        constructions = parts[cols.index("constructions")]
        conllu = parts[cols.index("conllu")]
        model_name = parts[cols.index("model_name")]
        if not (dependency == constructions == conllu):
            continue
        if dependency == "True":
            syntax = "True"
        else:
            syntax = "False"
        output.append("\t".join([sid, model_name, bleu, bert_f1, from_bible, lexicon, syntax]))

with open("stats_data.tab","w",encoding="utf8") as out_f:
    out_f.write("sid\tmodel_name\tBLEU\tBERTScore_F1\tbible\tlexicon\tsyntax\n")
    out_f.write("\n".join(output))
