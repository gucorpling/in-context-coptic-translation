import os, sys, re, codecs
from glob import glob
from collections import defaultdict


_unicode_escape_re = re.compile(r'\\\\u([0-9a-fA-F]{4})')

def decode_unicode_escapes_once(s):
    # Normalize escaped unicode escapes: \\uXXXX → \uXXXX
    s = _unicode_escape_re.sub(r'\\u\1', s)
    # Decode Unicode escapes once
    return codecs.decode(s, 'unicode_escape')

folder = "Gemma 27b-it"
folder = "Gemma 12b-it"
folder = "GPT-4.1"
folder = os.path.abspath(folder)

tsv_files = glob(os.path.join(folder,"ranked*.tsv"))
tsv_files = [f for f in tsv_files if "seed" not in f]

baseline_scores = {}

data = defaultdict(lambda: defaultdict(dict))
for f in tsv_files:
    # ranked_results_gemma-3-27b-it_baseline_dev etc. condition -> "baseline"
    condition = re.search(r'.*(it|14)_(.*)_dev\.tsv', os.path.basename(f)).group(2)
    for l, line in enumerate(open(f).read().strip().split("\n")):
        if l == 0: # Header
            cols = line.split("\t")
            continue
        parts = line.split("\t")
        # Convert parts[0] from '\u2c8f \u2c9b\u03e5\u03eb' to proper utf8
        # Also handle cases with double backslashes like  "said, \\u201cHe "
        parts[0] = decode_unicode_escapes_once(parts[0])
        data[condition][parts[0]] = {cols[i]: parts[i] for i in range(len(cols))}
        if condition == "baseline":
            baseline_scores[parts[0]] = float(parts[cols.index("BERTScore_F1")])

new_headers = ["source","reference"]
for i, sent in enumerate(baseline_scores):
    out_row = []
    for condition in sorted(data,key=lambda x: x!="baseline"):
        if condition == "baseline":
            # Get source	reference	predictions
            for key in ["source","reference"]:
                out_row.append(data[condition][sent][key])
        elif condition in ["conllu", "constructions","dependency"]:
            continue
        if i == 0:
            new_headers.append(f"{condition}_F1")
            new_headers.append(f"{condition}_F1_diff")
            new_headers.append(f"{condition}_pred")
        current = float(data[condition][sent]["BERTScore_F1"])
        out_row.append(f"{current:.4f}")
        baseline = baseline_scores[sent]
        diff = current - baseline
        out_row.append(f"{diff:.4f}")
        out_row.append(decode_unicode_escapes_once(data[condition][sent]["predictions"]).replace("\n"," ").replace("\t"," ").strip())

    if i == 0:
        print("\t".join(new_headers))
    print("\t".join(out_row))#.replace("\\n","").replace("\\",""))


