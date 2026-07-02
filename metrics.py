"""
translation_metrics.py
--------------------------------
Utility functions to evaluate translation outputs across multiple metrics:
- BLEU
- COMET
- chrF-S
- WER
- BERTScore
- chrF++

Model and dataset are modular so that future models (e.g., GPT-4o, GPT-5) can be plugged in easily.
"""
# TODO : Refactor and cleanup the metrics implementation
import pandas as pd
import numpy as np
import sacrebleu
import evaluate
from jiwer import wer
import logging

import util

logging.getLogger(__name__).setLevel(logging.INFO)

class Metrics:
    bert_metric = evaluate.load("bertscore")

    def __init__(self):
        pass

    # ====================================================
    # 2️⃣ BLEU
    # ====================================================
    # Sacrebleu is an implementation of BLEU
    @staticmethod
    def compute_bleu(predictions, references):
        """Compute corpus-level BLEU score."""
        if len(predictions) == 0 or len(predictions) != len(references):
            raise ValueError("Predictions and references must be non-empty and of equal length.")
        # Removing empty strings
        # Convert references to seq of seq
        references = [[reference] for reference in references]
        predictions = list(filter(lambda x: x != "", predictions))
        bleu = sacrebleu.corpus_bleu(predictions, references)
        return {"BLEU": bleu.score}


    # ====================================================
    # 3️⃣ COMET
    # ====================================================
    @staticmethod
    def compute_comet(predictions, references, sources, model_id="unbabel/wmt22-comet-da", batch_size=128):
        """Compute COMET metric with batching for large datasets."""
        comet_metric = evaluate.load("comet", model_id=model_id)

        preds = ["" if p is None else str(p) for p in predictions]
        refs = ["" if r is None else str(r) for r in references]
        srcs = ["" if s is None else str(s) for s in sources]

        all_seg_scores = []
        for i in range(0, len(preds), batch_size):
            chunk_preds = preds[i:i + batch_size]
            chunk_refs = refs[i:i + batch_size]
            chunk_srcs = srcs[i:i + batch_size]
            # https://github.com/huggingface/evaluate/issues/673 COMET needs num_workers
            out = comet_metric.compute(predictions=chunk_preds, references=chunk_refs, sources=chunk_srcs)
            if "scores" in out:
                all_seg_scores.extend(out["scores"])
            elif "score" in out:
                all_seg_scores.extend([float(out["score"])] * len(chunk_preds))
        return {"COMET": float(np.mean(all_seg_scores))}

    # ====================================================
    # 4️⃣ chrF-S (sentence-level average)
    # ====================================================
    @staticmethod
    def compute_chrf_s(predictions, references, char_order=6, word_order=2):
        """Compute chrF-S: average of sentence-level chrF scores."""
        chrf = evaluate.load("chrf")
        scores = []
        for pred, ref in zip(predictions, references):
            result = chrf.compute(predictions=[pred], references=[[ref]], char_order=char_order, word_order=word_order)
            scores.append(result["score"] / 100.0)
        return {"chrF-S": float(np.mean(scores))}


    # ====================================================
    # 5️⃣ WER
    # ====================================================
    @staticmethod
    def compute_wer_metric(predictions, references):
        """Compute average Word Error Rate (WER)."""
        total = sum(wer(ref, pred) for ref, pred in zip(references, predictions))
        return {"WER": total / len(predictions)}


    # ====================================================
    # 6️⃣ BERTScore
    # ====================================================
    @staticmethod
    def compute_bertscore(predictions, references, lang="en"):
        """Compute BERTScore precision, recall, and F1."""
        results = Metrics.bert_metric.compute(predictions=predictions, references=references, lang=lang)
        return {
            "BERTScore_P": float(np.mean(results["precision"])),
            "BERTScore_R": float(np.mean(results["recall"])),
            "BERTScore_F1": float(np.mean(results["f1"]))
        }

    @staticmethod
    def compute_bertscores_for_samples(predictions, references, lang="en"):
        """Compute BERTScore precision, recall, and F1."""
        results = Metrics.bert_metric.compute(predictions=predictions, references=references, lang=lang)
        return {
            "BERTScore_P": results["precision"],
            "BERTScore_R": results["recall"],
            "BERTScore_F1": results["f1"],
        }


    # ====================================================
    # 7️⃣ chrF++
    # ====================================================
    @staticmethod
    def compute_chrfpp(predictions, references):
        """Compute corpus-level chrF++ score."""
        chrf_metric = evaluate.load("chrf")
        results = chrf_metric.compute(predictions=predictions, references=references, word_order=2)
        return {"chrF++": results["score"]}

    @staticmethod
    def compute_meteor(predictions, references):
        """Compute corpus-level METEOR score."""
        meteor_metric = evaluate.load("meteor")
        results = meteor_metric.compute(predictions=predictions, references=references)
        return results


    # ====================================================
    # 🧩 Master Wrapper Function (run all)
    # ====================================================
    @staticmethod
    def evaluate_sample(prediction, reference):
        if prediction == None:
            prediction = [""]
        # print(f"Prediction: {prediction}")
        prediction = [prediction]
        reference = [reference]

        return {
            **Metrics.compute_bleu(prediction, reference),
            # FIXME: COMET currently doesn't work with Apple Silicon
            # **Metrics.compute_comet(predictions, references, sources),
            **Metrics.compute_chrf_s(prediction, reference),
            **Metrics.compute_wer_metric(prediction, reference),
            **Metrics.compute_bertscore(prediction, reference),
            **Metrics.compute_chrfpp(prediction, reference),
            **Metrics.compute_meteor(prediction, reference),
        }

    @staticmethod
    def evaluate_sample_bleu(prediction, reference):
        if prediction == None:
            prediction = [""]
        # print(f"Prediction: {prediction}")
        prediction = [prediction]
        reference = [reference]

        return {
            **Metrics.compute_bleu(prediction, reference),
        }


    @staticmethod
    def evaluate_all_metrics(predictions, references):
        """Run all metrics and return a summary dict."""

        # clean predictions with util.clean_string
        predictions = [util.clean_string(prediction) for prediction in predictions]

        return {
            **Metrics.compute_bleu(predictions, references),
            # **Metrics.compute_comet(predictions, references, sources),
            **Metrics.compute_chrf_s(predictions, references),
            **Metrics.compute_wer_metric(predictions, references),
            **Metrics.compute_bertscore(predictions, references),
            **Metrics.compute_chrfpp(predictions, references),
            **Metrics.compute_meteor(predictions, references),
        }
