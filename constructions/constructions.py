import io, os, sys, re
import logging

from argparse import ArgumentParser

import pandas as pd
from depedit import DepEdit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

rephrase = {
    "person=1": "first person",
    "person=2": "second person",
    "person=3": "third person",
    "number=Sing": "singular",
    "number=Plur": "plural",
    "third person+singular": "he/she/it",
    "third person+plural": "they",
    "first person+singular": "I",
    "first person+plural": "we",
    "second person+singular": "you (singular)",
    "second person+plural": "you (plural)",
    "PROⲁⲛⲟⲕ": "I",
    "PROⲛⲧⲟⲟⲩ": "they",
    "PROⲛⲧⲟϥ": "he/it",
    "PROⲛⲧⲟⲥ": "she/it",
    "PROⲁⲛⲟⲛ": "we",
    "PROⲛⲧⲟk": "you (masculine singular)",
    "PROⲛⲧⲟ": "you (feminine singular)",
    "PROⲛⲧⲱⲧⲛ": "you (plural)",

    "they VERBs": "they VERB",
    "we VERBs": "we VERB",
    "you VERBs": "you VERB",
    "let we": "let us",
    "let I": "let me",
    "let he": "let him",
    "let she": "let her",
    "let they": "let them",
    "make we": "make us",
    "make I": "make me",
    "make he": "make him",
    "make she": "make her",
    "make they": "make them",
    "do he": "does he",
    "do she": "does she",
    "I VERBs": "I VERB",
    " Fem ": " feminine ",
    " Masc ": " masculine ",
}

construction_map = {
    "APST": f'''The past auxiliary $2 is used with the person=$4 number=$5 subject pronoun $3 to express a past tense for the verb $6 (person=$4+number=$5 VERBed).''',
    "ANEGPST": f'''The negative past auxiliary $2 is used with the person=$4 number=$5 subject pronoun $3 to express a negated past tense for the verb $6 (person=$4+number=$5 did not VERB).''',
    "AAOR": f'''The aorist auxiliary $2 is used with the person=$4 number=$5 subject pronoun $3 and the verb $6 to express a general fact (person=$4+number=$5 (always/generally) does VERB). There is often no need to translate a word like "always".''',
    "ANEGAOR": f'''The negative aorist auxiliary $2 is used with the person=$4 number=$5 subject pronoun $3 and the verb $6 to express a negated general fact (person=$4+number=$5 (generally) does not VERB). There is often no need to translate a word like "generally".''',
    "AJUS": f'''The jussive auxiliary $2 is used with the person=$4 number=$5 subject pronoun $3 and the verb $6 to express an exhortation (let person=$4+number=$5 VERB, or shall person=$4+number=$5 VERB?).''',
    "ACAUS": f'''The causative $2 is used with the pronoun $3 the verb $5 to express "make PRO$4 VERB".''',
    "ANEGOPT": f'''The negative optative auxiliary $2 is used with the person=$4 number=$5 subject pronoun $3 and the verb $6 to express a negated wish or hope (may person=$4+number=$5 not VERB).''',
    "APREC": f'''The precursive auxiliary $2 is used with the person=$4 number=$5 subject pronoun $3 to express a subordinate "after" clause with the verb $6 (after person=$4+number=$5 VERBed).''',
    "AOPT_PPERS": f'''The fused optative auxiliary $2 which contains a person=$3 number=$4 pronoun is used with the verb $5 to express a wish or hope (may person=$3+number=$4 VERB).''',
    "ACOND_PPERS": f'''The fused conditional auxiliary $2 which contains a person=$3 number=$4 pronoun is used with the verb $5 to express a condition (if person=$3+number=$4 VERBs).''',
    "N_ACOND": f'''The conditional auxiliary $2 is used with the subject phrase headed by $3 and the verb $4 to express a condition (if NOUN VERBs).''',
    "N_APST": f'''The past auxiliary $2 is used with the subject phrase headed by $3 to express a past tense for the verb $4 (NOUN VERBed).''',
    "N_ANEGPST": f'''The negative past auxiliary $2 is used with the subject phrase headed by $3 to express a negated past tense for the verb $4 (NOUN did not VERB).''',
    "N_AOR": f'''The aorist auxiliary $2 is used with the subject phrase headed by $3 and the verb $4 to express a general fact (NOUN (always/generally) does VERB). There is often no need to translate a word like "always".''',
    "N_ANEGAOR": f'''The negative aorist auxiliary $2 is used with the subject phrase headed by $3 and the verb $4 to express a negated general fact (NOUN (generally) does not VERB). There is often no need to translate a word like "generally".''',
    "N_AOPT": f'''The optative auxiliary $2 is used with the subject phrase headed by $3 and the verb $4 to express a wish or hope (may NOUN VERB).''',
    "N_ANEGOPT": f'''The negative optative auxiliary $2 is used with the subject phrase headed by $3 and the verb $4 to express a negated wish or hope (may NOUN not VERB).''',
    "VSTAT": f'''The stative verb form $2 (for the lemma $3) is used with the subject $4 to express a state - note that for a transitive verb, this will have passive meaning (ⲕⲱⲧ 'build', stative ⲕⲏⲧ 'is built'), but for intransitives it expresses a state of being (ϩⲗⲟϭ 'become sweet', stative ϩⲟⲗϭ 'be sweet'), so we need to consider whether $3 is transitive.''',
    "VSTATO": f'''The special stative verb form ⲟ (for the lemma $3), usually followed by a state introduced by ⲛ is used to express the meaning become STATE, be STATE (here the state is expressed by $5 for the subject $4).''',
    "Counter_FUT": f'''The combination of the future auxiliary $2 with the counterfactual preterit $3 is used together with the predicate $4 to express a counterfactual meaning (would have VERBed).''',
    "FUT_pron": f'''The future auxiliary $2 is used with the predicate $3 and the subject pronoun $4 to express a future tense (PRO$5 will PREDICATE).''',
    "FUT": f'''The future auxiliary $2 is used with the predicate $3 to express a future tense (will PREDICATE).''',
    "PET": f'''The construction with the $3 singular definite article $2 followed by the relativizer $4 ($2 $4) means "the one who...".''',
    "NET": f'''The construction with the plural definite article $2 followed by the relativizer $4 ($2 $4) means "the ones who...".''',
    "NQI": f'''The predicate $2 has a postponed subject $3 introduced by $4. $5 VERB literally means "person=$7+number=$6 VERB, that is $3...", but often the postponed subject (here $3 etc.) can simply be translated as the subject of the predicate.''',
    "DISLOC": f'''The dislocated element $3 is a repeated reference to the pronoun dependent of the predicate $2. There is often no need to translate the pronominal mention of the same argument.''',
    "DBLNEG": f'''The doubled circum-negation with $2 ... $4 $3 is used to negate the predicate $4, its overall meaning is the same as a simple negation, so we should translate this as not + $4.''',
    "IMPER": f'''The verb $2 is being used as an imperative.''',
    "PLUR": f'''the phrase headed by $2 $3 is plural.''',
    "MMOS": f'''The empty expletive pronoun phrase ⲙⲙⲟ $3 is used redundantly with the complement clause predicate $4 and does not need to be translated. This construction is commonly used with speech or thought verbs, literally akin to Eng. "say (it) that...".''',
    "JOOS": f'''The empty expletive pronoun $3 is used redundantly with the complement clause predicate $4 and does not need to be translated. This construction is commonly used with speech or thought verbs, literally akin to Eng. "say (it) that...".''',
    "NAEIAT": f'''The construction with ⲛⲁⲉⲓⲁⲧ means "blessed is/are...", here used with the subject $2.''',
    "AHRO": f'''The construction with ⲁϩⲣⲟ means "why...", here used with the subject $2, meaning "why do PRO$3".''',
    "NAME":f'''$2 is a proper noun, its transliteration in English letters is @$2@ (it may also be better to translate it to a conventional form).''',
    "MNTREF": f'''The prefix $3 in $2 forms an abstract noun (like Eng. ness) from an agent noun (like Eng. er) derived from $4, akin to English VERB-er-ness.''',
    "MNTAT": f'''The prefix $3 in $2 forms an abstract noun (like Eng. ness) with a negation (like Eng. -less) derived from $4, akin to English X-less-ness.''',
    "MNT": f'''The prefix $3 in $2 forms an abstract noun (like Eng. ness) derived from $4, akin to English X-ness.''',
    "REF": f'''The prefix $3 in $2 forms an agent noun (like Eng. -er) derived from $4, akin to English VERB-er.''',
    "AT": f'''The prefix $3 in $2 forms a negative agent noun derived from $4, akin to English 'X-less one'.''',
}

script_dir = os.path.dirname(os.path.abspath(__file__))

class ConstructionExplainer:

    def __init__(self, config_file="cop_mt_constructions.ini"):
        # Initialize DepEdit with the given configuration file so we don't have to reload it for each call
        if not os.path.isabs(config_file):  # Assume relative paths are relative to this script, not working directory
            config_file = os.path.join(script_dir, config_file)
        self.deped = DepEdit(config_file=config_file)

    @staticmethod
    def transliterate_name(template):
        """
        Edit a template containing an item to transliterate between @ signs

        :param template: a template string containing some Coptic name, e.g. "... @ϣⲉⲛⲟⲩⲧⲉ@..."
        :return: template with the string transliterated assuming title case, e.g. "... Shenoute ..."
        """
        charmap = {"ⲁ": "a", "ⲃ": "b", "ⲅ": "g", "ⲇ": "d", "ⲉ": "e", "ⲍ": "z", "ⲏ": "e", "ⲑ": "th", "ⲓ": "i", "ⲕ": "k",
                   "ⲗ": "l", "ⲙ": "m", "ⲛ": "n", "ⲝ": "x", "ⲟ": "o", "ⲡ": "p", "ⲣ": "r", "ⲥ": "s", "ⲧ": "t", "ⲩ": "u",
                   "ⲫ": "ph", "ⲭ": "kh", "ⲯ": "ps", "ⲱ": "o", "ϣ": "sh", "ϥ": "f", "ϧ": "kh", "ϩ": "h", "ϫ": "j",
                   "ϭ": "ch","ϯ": "ti"}
        to_translit = re.findall("(@.*?@)", template)
        replacements = []
        for foreign in to_translit:
            translit = []
            for c in foreign.lower():
                mapped = charmap[c] if c in charmap else c
                translit.append(mapped)
            replacements.append((foreign,"".join(translit).replace("@","").title()))

        for find, replace in replacements:
            template = template.replace(find, replace)

        return template

    def get_construction_prompts(self, conllu, with_sent=False):
        """
        Generate construction prompts from a CoNLL-U formatted string using DepEdit.

        :param conllu: Input CoNLL-U formatted string
        :param config_file: Path to the DepEdit configuration file
        :return: List of prompts for each sentence, each prompt is a string with one construction explanation per line, or an empty string if no constructions found
        """
        # Replace hyphen separator in MSeg with "%" to avoid conflicting with construction hyphen notation
        escaped = []
        for l in conllu.split("\n"):
            if "\t" in l:
                cols = l.split("\t")
                cols[-1] = cols[-1].replace("-","%")
                l = "\t".join(cols)
            escaped.append(l)
        conllu = self.deped.run_depedit("\n".join(escaped))
        prompts = []
        n_cons = []
        # TODO use pyconll/organize code
        for sent in conllu.strip().split("\n\n"):
            sent_prompts = []
            n_sent_cons = 0
            for line in sent.split("\n"):
                if line.startswith("# text = ") and with_sent:
                    sent_prompts.append(line.split("=")[1].strip())
                if "\t" in line:
                    cols = line.split("\t")
                    if "." in cols[0] or "-" in cols[0]:
                        continue
                    if "MT=" in cols[-1]:
                        m = re.search(r'MT=([^|]+)', cols[-1])
                        if m is not None:
                            constructions = m.group(1)
                            for construction in constructions.split(","):
                                parts = construction.split("-")
                                if parts[0] in construction_map:
                                    template = construction_map[parts[0]]
                                    n_sent_cons += 1
                                    for i in range(1, len(parts)+1):
                                        template = template.replace(f"${i}", parts[i-1])
                                    if "@" in template:
                                        template = self.transliterate_name(template)
                                    for key, val in rephrase.items():
                                        template = template.replace(key, val)
                                    sent_prompts.append(template.replace("%","-"))
            n_cons.append(n_sent_cons)
            prompts.append(r"\n".join(sent_prompts).strip())
        logger.info(f"Generated construction prompts for {len(prompts)} sentences.")
        n_cons = pd.Series(n_cons)
        logger.info(f"Construction counts summary:\n{n_cons.describe()}")
        return prompts


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-i', '--input', action='store', default="example.conllu", help='input conllu file')
    parser.add_argument('-c', '--config', action='store', default="cop_mt_constructions.ini", help='DepEdit config file')
    parser.add_argument('-p', '--print', action='store_true', help='Print prompts for inspection instead of logging')
    parser.add_argument('-s', '--sent', action='store_true', help='Also print the original sentence before each prompt for inspection')
    options = parser.parse_args()

    explainer = ConstructionExplainer(config_file=options.config)
    with io.open(options.input, encoding="utf8") as f:
        conllu  = f.read()
    prompts = explainer.get_construction_prompts(conllu, with_sent=options.sent)
    for i, sent_prompt in enumerate(prompts):
        if options.print:
            print(f"Sentence {i+1} constructions:")
            print(sent_prompt.replace("\\n","\n"))
            print()
        else:
            logger.debug(f"Sentence {i+1} constructions:")
            logger.debug(sent_prompt)


