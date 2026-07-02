# Take .conllu file and parse dependencies into plain English

from collections import Counter
import pyconll

# map dependency relationship markers to plain English
# https://universaldependencies.org/u/dep/index.html 
deprel_to_english = {
    "nsubj": "nominal subject",
    "obj": "object",
    "iobj": "indirect object",
    "csubj": "clausal subject",
    "ccomp": "clausal complement",
    "xcomp": "open clausal complement",
    "obl": "oblique nominal",
    "vocative": "vocative",
    "expl": "expletive",
    "dislocated": "dislocated element", 
    "advcl": "adverbial clause modifier",
    "advmod": "adverbial modifier",
    "discourse": "discourse element", 
    "aux": "auxiliary",
    "cop": "copula",
    "mark": "marker",
    "nmod": "nominal modifier",
    "appos": "appositional modifier",
    "nummod": "numeric modifier",
    "acl": "adnominal clause",
    "acl:relcl": "relative adnominal clause",
    "amod": "adjectival modifier",
    "nmod:poss": "possessive nominal modifier",
    "det": "determiner",
    "clf": "classifier",
    "case": "case marking",
    "conj": "conjunct",
    "cc": "coordinating conjunction",
    "fixed": "fixed multiword expression",
    "flat": "flat expression",
    "list": "list",
    "parataxis": "parataxis",
    "compound": "compound",
    "orphan": "orphan",
    "goeswith": "goes with",
    "reparandum": "overridden disfluency",
    "punct": "punctuation",
    "root": "root",
    "dep": "unspecified dependency"
}

collapse_deprel_to_english = {
    "nsubj": "subject",
    "csubj": "subject",
    "obj": "object",
    "iobj": "object",
    "ccomp": "complement",
    "xcomp": "complement",
    "obl": "oblique nominal",
    "vocative": "vocative",
    "expl": "expletive",
    "advcl": "modifier",
    "advmod": "modifier",
    "discourse": "discourse element", 
    "aux": "auxiliary",
    "cop": "copula",
    "mark": "marker",
    "nmod": "modifier",
    "appos": "modifier",
    "nummod": "modifier",
    "nmod:poss": "possessive",
    "acl": "clause",
    "amod": "modifier",
    "clf": "classifier",
    "case": "case marking",
    "conj": "conjunct",
    "cc": "coordinating conjunction",
    "fixed": "fixed multiword expression",
    "flat": "flat expression",
    "compound": "compound",
    "root": "root",
    "dep": "unspecified dependency"
}

upos_content = {"NOUN", "VERB", "PROPN", "ADP", "ADV"}

upos_participants_plus = {"NOUN", "VERB", "PROPN", "ADP", "ADV", "PRON", "AUX", "DET", "NUM"}

upos_other = {"NOUN", "VERB", "PROPN", "ADP", "ADV", "PRON", "AUX", "DET", "NUM", "CCONJ", "SCONJ"}

upos_blacklist = {"PUNCT", "SYM", "X"}

dup_ordinal = {
    "1": "the first",
    "2": "the second",
    "3": "the third",
    "4": "the fourth",
    "5": "the fifth",
}

dup_subscript= {
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
}

def explain_dependencies(sentences, duplicate_setting = None, collapse = False, upos_setting = None):
    # translate dependencies to plain English
    # deprel is the dependency relation of the current token to the HEAD (or ROOT iff HEAD = 0)

    explanations = {}
    all_exp = []

    # collapsed mappings or not
    mapping = collapse_deprel_to_english if collapse else deprel_to_english

    for sentence in sentences:
        sent_explanations = []
        sent_id = sentence.id
        # TODO:
        if duplicate_setting:
            forms = [t.form for t in sentence if "-" not in str(t.id)]
            form_counts = Counter(forms)
            duplicates = {f for f, c in form_counts.items() if c > 1}
            seen_counts = {f: 0 for f in duplicates}
            count_index = {}

            for token in sentence:
                if "-" not in str(token.id):
                    token_id = int(token.id)
                    form = token.form
                    if form in duplicates:
                        seen_counts[form] += 1 
                        count_index[token_id] = seen_counts[form]
                    else:
                        count_index[token_id] = 0

        for token in sentence:
            if "-" in str(token.id):
                continue
            token_id = int(token.id)
            form = token.form
            head_id = int(token.head)

            if upos_setting == "content" and token.upos not in upos_content:
                continue
            if upos_setting == "plus" and token.upos not in upos_participants_plus:
                continue
            if upos_setting == "other" and token.upos not in upos_other:
                continue
            if token.upos in upos_blacklist:
                continue

            if duplicate_setting and form in duplicates:
                if duplicate_setting == "ordinal":
                    form_display = f"{dup_ordinal.get(str(count_index[token_id]), '')} {form}"
                elif duplicate_setting == "subscript":
                    form_display = f"{form}{dup_subscript.get(str(count_index[token_id]), '')}"
            else:
                form_display = form

            if head_id == 0:
                head = "ROOT"
            else:
                # head is the sentence[head_id]["form"]
                head_token = sentence[str(head_id)]
                head = head_token.form
                if duplicate_setting and head in duplicates:
                    if duplicate_setting == "ordinal":
                        head = f"{dup_ordinal.get(str(count_index[head_id]), '')} {head}"
                    elif duplicate_setting == "subscript":
                        head = f"{head}{dup_subscript.get(str(count_index[head_id]), '')}"
                
            deprel = token.deprel
            deprel_english = mapping.get(deprel)
            if deprel_english is not None:
                if deprel_english == "root":
                    plain_text = f"{form_display} is the {deprel_english}."
                else:
                    plain_text = f"{form_display} is the {deprel_english} of {head}."
                sent_explanations.append(plain_text)

        explanations[sent_id] = {
            "explanations": sent_explanations
        }
        # collapse each sentence’s explanations into a single string
        all_exp.append(" ".join(sent_explanations))
  
    return all_exp

def get_llm_input(filepath, duplicate_setting=None, collapse=False, upos_setting=None,
        dep_setting = None):
    
    corpus = pyconll.load_from_file(filepath)
    with open(filepath,encoding="utf8") as f:
        conllu_sentences = f.read().split("\n\n")

    prompts=[]
    explanations = explain_dependencies(corpus, duplicate_setting, collapse, upos_setting)
    return explanations
    

# main calls
if __name__ == "__main__":
    #filepath = "data/ud_coptic/ud_coptic_dev_pred.conllu"
    filepath = "parsing/data/sample_bohairic_ud.conllu"
    # corpus = pyconll.load_from_file(filepath)
    # with open(filepath,encoding="utf8") as f:
    #     conllu_sentences = f.read().split("\n\n")
    prompt = get_llm_input(filepath, "subscript", True, "other")
    print(prompt)

    
