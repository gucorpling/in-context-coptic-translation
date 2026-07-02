
import argparse
import configparser
import logging
import pprint as pp

import pyconll
from datasets import Dataset

import lexicon
import util
from constructions.constructions import ConstructionExplainer
from lexicon import Lexicon
import ud_parsing as parsing

from types import SimpleNamespace

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LRMTDataConfig(SimpleNamespace):
    pass

def load_parallel_sentences(cop_file, eng_file):
    # TODO fix loading to avoid using f.read().splitlines() - https://huggingface.co/docs/datasets/v4.2.0/nlp_load

    with open(cop_file, 'r', encoding='utf-8') as f:
        cop_sentences = f.read().splitlines()

    with open(eng_file, 'r', encoding='utf-8') as f:
        eng_sentences = f.read().splitlines()

    return Dataset.from_dict({
        "source" : cop_sentences,
        "reference" : eng_sentences,
    })

def prompt_for_translation(cop_sentence, lexicon_text="x", dependency_text="", conllu_text="", construction_text=""):
    """
    Instruction based on WMT 25
    ref: https://gist.github.com/Abhishek-P/62c176bd473cb5fed38752f9fc08cfb5
    :param conllu_text:
    :param cop_sentence:
    :param lexicon_text:
    :param dependency_text:
    :return:
    domain_description = "is in the domain of religious texts"
    domain_instruction = "Translate with attention to religious terminology and context. "
    """
    source_language = "Coptic"
    target_language = "English"
    target_region = "United States"
    target_language_code = "en_US"
    # FIX construction, dependency ordering w.r.t to conllu dumping
    prompt = (
        fr"You are a professional {source_language}-to-English translator tasked with providing translations suitable for use"
        fr" in {target_region} ({target_language_code}). Your goal is to accurately convey the meaning and nuances of the"
        fr" original {source_language} text while adhering to {target_language} grammar, vocabulary, and cultural sensitivities."
        fr"Produce only the {target_language} translation, without any additional explanations or commentary."
        fr"Please translate the following {source_language} text into {target_language} ({target_language_code}):\n{cop_sentence}."
        # fr"The original Coptic text {domain_description}. {domain_instruction}"
        #fr" Retain the paragraph breaks (double new lines) from the input text."
        # TODO refactor lexicon text to match dependency text style
        fr"{lexicon_text}"
        fr"{conllu_text}"
        fr"{dependency_text}"
        fr"{construction_text}"
        fr"Using all the information provided above, now please translate the sentence into {target_language}({target_language_code})."
        fr"Remember your source sentence is: {cop_sentence}."
        fr"The {target_language} translation is:"

    )
    return prompt

def get_prompt_dataset(cop_file, eng_file, conllu_file, lexicon_file, config=None):
    prompt_func = prompt_for_translation
    dataset = load_parallel_sentences(cop_file, eng_file)
    # The four files are mandatory - cop_file, eng_file, lexicon_file, conllu_file
    assert cop_file
    assert eng_file
    assert lexicon_file
    assert conllu_file

    logger.info(f"Using components config {pp.pformat(config)}")

    """
    components config
    
    dependency
        "enabled" : False | True
        "duplicates": "ordinal"|"subscript"
        "collapse": "True|False"
        "upos_setting": "content"|"plus"|"other"
        
    # TODO move this up her
    conllu.enabled : False | True
    conllu.setting : "first" | "last" | "only"
    lexicon. :
        enabled : False | {
        entry_langs : ["German"?, "French"?, "English"?]
        pos_tags: [Lexicon.Pos],
        grammatical_info: False, 
        max_entries: 0..100,
        max_senses_per_entry : 0..100
        dedup_senses: False | True
        }    
    }
    """

    if config["dependency.enabled"] == 'True':
        dependency_texts = parsing.get_llm_input(conllu_file, config["duplicates"], config["collapse"],
                                                 config["upos_setting"])
        dataset = dataset.add_column("dependency_text", dependency_texts)

    if config['conllu.enabled'] == "True":
        conllu = pyconll.load_from_file(conllu_file)
        def clean_conllu_metadata(sentence):
            sentence._meta = dict()
            return sentence
        conllu_texts = [clean_conllu_metadata(sentence).conll() for sentence in conllu]
        dataset = dataset.add_column("conllu_text", conllu_texts)

    if config["constructions.enabled"] == 'True':
        # TODO Add sample level fields based on OrigLang

        explainer = ConstructionExplainer()
        with open(conllu_file, "r", encoding="utf8") as f:
            conllu_data = f.read()
        construction_texts = explainer.get_construction_prompts(conllu_data)
        dataset = dataset.add_column("construction_text", construction_texts)

    if config["lexicon.enabled"] == 'True':
        # TODO use Mseg as fall back for retrieval
        lexicon_retriever = Lexicon(lexicon_file)
        # TODO: move to lexicon module
        def get_lexicon_texts_for_conllu(conllu_file):
            conllu_sentences = pyconll.load_from_file(conllu_file)
            def lexicon_text(sentence):
                forms = [token.form for token in sentence]
                lemmas = [token.lemma for token in sentence]
                uposes = [token.upos for token in sentence]
                def get_mseg(token):
                    if "MSeg" in token.misc:
                        return list(token.misc["MSeg"])[0]
                    else:
                        return None

                msegs = [get_mseg(token) for token in sentence]
                return lexicon_retriever.get_lexicon_prompt_text(forms, lemmas, uposes, msegs, config)
            return [lexicon_text(sentence) for sentence in conllu_sentences]

        lexicon_texts = get_lexicon_texts_for_conllu(conllu_file)
        dataset = dataset.add_column("lexicon_text", lexicon_texts)

    # FIX: flagging for prompt function
    dataset = dataset.map(
        lambda x: {"prompt": prompt_func(x["source"], x.get("lexicon_text", ""),
            fr"The dependency information for the sentence is:\n{x['dependency_text']}\n" if "dependency_text" in x else "",
            fr"The raw conllu data for the sentence is in the CONLL-U format: \n{x['conllu_text']}\n" if "conllu_text" in x else "",
            fr"The information about specific constructions in the sentence are:\n{x['construction_text']}\n" if "construction_text" in x else "",
        )}
    )
    return dataset


# def get_prompt_dataset(cop_file, eng_file, conllu_file, prompt_func):
#     dataset = load_parallel_sentences(cop_file, eng_file, conllu_file).map(lambda x: {"prompt": prompt_func(x["source"], x["conllu"])})
#     return dataset

if __name__ == "__main__":
    lexicon_file = "data/lexicon.xml"
    # cop_file = "data/ud_coptic/ud_coptic_dev.txt"
    # eng_file = "data/ud_coptic/ud_coptic_dev_en.txt"
    # conllu_file = "data/ud_coptic/ud_coptic_dev_pred.conllu"
    # output_file = f"data/temp/prompts_ud_coptic_dev.tsv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--cop-sentences", type=str, required=True)
    parser.add_argument("--eng-sentences", type=str, required=True)
    parser.add_argument("--conllu", type=str, default=None)
    parser.add_argument("--use_lexicon", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    cop_file = args.cop_sentences
    eng_file = args.eng_sentences
    conllu_file = args.conllu
    output_file = args.out
    print(args)
    config = util.get_components_config("configs/dev.components.ini")
    dataset = get_prompt_dataset(cop_file, eng_file, conllu_file, lexicon_file, config)
    dataset = util.encode_dataset(dataset)
    dataset.to_csv(output_file, sep="\t", index=False)

