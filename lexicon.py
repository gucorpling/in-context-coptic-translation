"""
This is python script to process the Coptic dictionary XML file and prepare it for augmentation for Translation.
"""
import codecs
import dataclasses
import logging
import argparse, configparser
from functools import reduce
from collections import Counter, OrderedDict

import xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

from enum import Enum
import json

import pyconll

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CopticDialect(Enum):
    S = "S"
    B = "B"

class DictionaryLangs(Enum):
    DE = "de"#, "German"
    EN = "en"#, "English"
    FR = "fr"#, "French"

    @staticmethod
    def values():
        return [lang.value for lang in DictionaryLangs]

# With Anthropic Claude - https://claude.ai/share/d5f21ec0-289d-4313-8307-b33c5bd4957e
coptic_grammar_indicators = {
    "Temporalis ⲛⲧⲉⲣⲉ-": "Temporal (when-clause) with ⲛⲧⲉⲣⲉ-",
    "Qualitativ": "Qualitative (stative verb form)",
    "Qualitativ?": "Qualitative? (uncertain)",
    "Imperativ": "Imperative",
    "Konjunktiv ⲛ(ⲧⲉ)-": "Subjunctive with ⲛ(ⲧⲉ)-",
    "Perfekt ⲁ-": "Perfect tense with ⲁ-",
    "Perfekt neg. ⲙⲡⲉ-": "Negative perfect with ⲙⲡⲉ-",
    "Perfekt II ⲉⲛⲧⲁ-": "Perfect II (relative perfect) with ⲉⲛⲧⲁ-",
    "Aorist ϣⲁ(ⲣⲉ)-": "Aorist with ϣⲁ(ⲣⲉ)-",
    "Aorist neg. ⲙⲉ(ⲣⲉ)-": "Negative aorist with ⲙⲉ(ⲣⲉ)-",
    "Imperfekt ⲛⲉ(ⲣⲉ)-": "Imperfect with ⲛⲉ(ⲣⲉ)-",
    "Futur I / II ⲛⲁ-": "Future I/II with ⲛⲁ-",
    "Futur III ⲉ(ⲣⲉ)-": "Future III with ⲉ(ⲣⲉ)-",
    "Futur III neg. ⲛⲛⲉ-": "Negative future III with ⲛⲛⲉ-",
    "Kompletiv neg. ⲙⲡⲁ(ⲧⲉ)-": "Negative completive with ⲙⲡⲁ(ⲧⲉ)-",
    "Konditionalis ⲉⲣϣⲁⲛ-": "Conditional with ⲉⲣϣⲁⲛ-",
    "Infinitiv": "Infinitive",
    "Kausativer Infinitiv ⲧⲣⲉ-": "Causative infinitive with ⲧⲣⲉ-",
    "Kausativer Imperativ ⲙⲁⲣⲉ-": "Causative imperative with ⲙⲁⲣⲉ-",
    "Negierter Imperativ ⲙⲡⲣ-": "Negative imperative with ⲙⲡⲣ-",
    "Imperativpräfix ⲁ-": "Imperative prefix ⲁ-",
    "Reflexivpronomen": "Reflexive pronoun",
    "Indefinitpronomen": "Indefinite pronoun",
    "Fragepronomen": "Interrogative pronoun",
    "1. Pers.": "1st person",
    "2. Pers. ": "2nd person",
    "3. Pers.": "3rd person",
    "Status pronominalis": "Pronominal state",
    "Status nominalis": "Nominal state",
    "Negationspartikel": "Negation particle",
    "Fragepartikel": "Question particle",
    "Genitivpartikel": "Genitive particle",
    "Relativkonverter": "Relative converter",
    "Umstandsatzkonverter": "Circumstantial converter",
    "Fokalisierungskonverter": "Focusing converter",
    "im negativen Bedingungssatz": "In negative conditional clause",
    "Negiertes Futur im Bedingungssatz ⲁⲛⲛⲉ- (B)": "Negated future in conditional clause with ⲁⲛⲛⲉ- (Bohairic)",
    "Negativpräfix zur Bildung von Substantiven f.": "Negative prefix for forming feminine nouns",
    "Negativpräfix zur Bildung von Adjektiven": "Negative prefix for forming adjectives",
    "Kopula": "Copula (linking verb)",
    "Hilfsverb": "Auxiliary verb",
    "Adjektivverb": "Adjectival verb",
    "Kompositverb": "Compound verb",
    "Kompositverb (?)": "Compound verb (uncertain)",
    "Suffixkonjugation": "Suffix conjugation",
    "Participium conjunctum": "Conjunctive participle",
    "Obsolete Relativform": "Obsolete relative form",
    "Unpersönlich": "Impersonal",
    "Frageadverb": "Interrogative adverb",
    "zur Bildung von Ortsangaben": "For forming place expressions",
    "Völkername": "Ethnic name / Name of a people",
    "Göttername": "Divine name / Name of a deity",
    "Ortsname": "Place name",
    "Titel": "Title",
    "Name einer Sache oder Institution": "Name of a thing or institution"
}

# class G(Enum):
#     Noun='Noun'
#     Adverb='Adverb'
#     Verb='Verb'
#     Preposition='Preposition'
#     Number='Number'
#     Particle='Particle'
#     PersonalPronoun='Personal Pronoun'
#     Conjunction='Conjunction'
#     DemonstrativePronoun='Demonstrative Pronoun'
#     Article='Article'
#     Possessive='Possessive'
#     PossessivePronoun='Possessive Pronoun'
#     InterrogativePron='Interrogative Pronoun'
#     AuxiliaryVerb='Auxiliary Verb'


class Pos(Enum):
    Noun = 'N', 'Noun'
    Adverb = 'ADV', 'Adverb'
    Verb = 'VERB', 'Verb'
    # TODO check if Vstat should be something else
    VSTAT = 'VSTAT', 'Verb'
    VBD = 'VBD', 'Verb'
    VIMP = 'VIMP', 'Verb'
    PREP = 'PREP', 'Preposition'
    NUM = 'NUM', 'Number'
    PTC = 'PTC', 'Particle'
    PPER = 'PPER', 'Personal Pronoun'
    CONJ = 'CONJ', 'Conjunction'
    PDEM = 'PDEM', 'Demonstrative Pronoun'
    ART = 'ART', 'Article'
    PPOS = 'PPOS', 'Possessive'  # Possessive article or prefix
    PPERO = 'PPERO', 'Possessive Pronoun'
    PINT = 'PINT', 'Interrogative Pronoun'
    A = "A", 'Auxiliary Verb'

    def __init__(self, tag, plain):
        self.tag = tag
        self.plain = plain


DEFAULT_LEXICON_RETRIEVAL_CONFIG = {
    "entry_langs" : DictionaryLangs.values(),
     "pos_tags" : sorted([pos.plain for pos in Pos]),
    "dedup_senses" : False,
    "grammatical_info" : False,
    "max_entries" : 100,
    "max_senses_per_entry" : 100
}

## POS Map
"""Copied from : https://github.com/KELLIA/dictionary/blob/d315fccbe58da755b7360e7a01bf460a9f94fa41/utils/dictionary_reader.py#L466"""
def new_pos_map(pos, subc, orthstring):
    """
    :param pos: string - Coptic Dictionary linguistic tag
    :param subc: string - Coptioc Dictionary linguistic subcategory
    :param orthstring: string
    :return: string - Mapped POS tag for simplification
    """

    # TODO abhip convert to pattern mactching for readability
    class Gram(Enum):
        EXISTENTIAL = "EXIST", "Existential"
        NEG = "NEG", "Negation"
        # TODO abhip adjective?
        C = "C", "Converter"

        def __init__(self, tag, plain):
            self.tag = tag
            self.plain = plain

    # utils.get_examples.format_pos

    pos = pos.replace('?', '')
    match (pos, subc, orthstring):
        # if pos == u"Subst." or pos == u"Adj." or pos == u"Nominalpräfix" or pos == u"Adjektivpräfix" \
        #     or pos == u"Kompositum":
        case ("Subst.", _, _) | ("Adj.", _, _) | ("Nominalpräfix", _, _) | ("Adjektivpräfix", _, _) | ("Kompositum", _, \
                                                                                                       _):
            return Pos.Noun.plain

        # elif u"Ausdruck der Nichtexistenz" in subc or u"Ausdruck des Nicht-Habens" in subc:
        case (_, subc, _) if subc is not None and (
                ("Ausdruck der Nichtexistenz" in subc) or ("Ausdruck des Nicht-Habens" in subc)
        ):
            return Gram.EXISTENTIAL.plain

        # elif pos == u"Adv.":
        case ("Adv.", _, _):
            return Pos.Adverb.plain

        # elif pos == u"Vb." or pos == u"unpersönlicher Ausdruck":
        case ("Vb.", _, _) | ("unpersönlicher Ausdruck", _, _):
            if subc == u"Qualitativ":
                return Pos.VSTAT.plain
            elif subc == u"Suffixkonjugation":
                return Pos.VBD.plain
            elif subc == u"Imperativ":
                return Pos.VIMP.plain
            elif orthstring is not None and (u"ⲟⲩⲛ-" in orthstring or u"ⲟⲩⲛⲧⲉ-" in orthstring):
                return Gram.EXISTENTIAL.plain
            else:
                return Pos.Verb.plain

        case (u"Präp.", _, _):
            return Pos.PREP.plain

        # elif pos == u"Zahlzeichen" or pos == u"Zahlwort" or pos == u"Präfix der Ordinalzahlen":
        case (u"Zahlzeichen", _, _) | (u"Zahlwort", _, _) | (u"Präfix der Ordinalzahlen", _, _):
            return Pos.NUM.plain

        # elif pos == u"Partikel" or pos == u"Interjektion" or pos == u"Partikel, enklitisch":
        case (u"Partikel", _, _) | (u"Interjektion", _, _) | (u"Partikel, enklitisch", _, _):
            return Pos.PTC.plain

        # elif pos == u"Selbst. Pers. Pron." or pos == u"Suffixpronomen" or pos == u"Präfixpronomen (Präsens I)":
        case (u"Selbst. Pers. Pron.", _, _) | (u"Suffixpronomen", _, _) | (u"Präfixpronomen (Präsens I)", _, _):
            return Pos.PPER.plain

        # elif pos == u"Konj.":
        case (u"Konj.", _, _):
            return Pos.CONJ.plain  # 'CONJ'

        # elif pos == u"Dem. Pron.":
        case (u"Dem. Pron.", _, _):
            return Pos.PDEM.plain  # "PDEM"

        # elif pos == u"bestimmter Artikel" or pos == u"unbestimmter Artikel":
        case (u"bestimmter Artikel", _, _) | (u"unbestimmter Artikel", _, _):
            return Pos.ART.plain  # 'ART'

        # elif pos == u"Possessivartikel" or pos == u"Possessivpräfix":
        case (u"Possessivartikel", _, _) | (u"Possessivpräfix", _, _):
            return Pos.PPOS.plain  # 'PPOS'
        # elif pos == u"Poss. Pron.":
        case (u"Poss. Pron.", _, _):
            return Pos.PPERO.plain  # 'PPERO'
        # elif pos == u"Interr. Pron.":
        case (u"Interr. Pron.", _, _):
            return Pos.PINT.plain  # 'PINT'
        # elif pos == u"Verbalpräfix":
        case (u"Verbalpräfix", _, _):
            if subc == u"Imperativpräfix ⲁ-" or subc == u"Negierter Imperativ ⲙⲡⲣ-":
                return Gram.NEG.plain  # 'NEG'
            if subc == u"im negativen Bedingungssatz" or subc == u"Perfekt II ⲉⲛⲧⲁ-":
                return 'NONE'
            else:
                return Pos.A.plain  # 'A'
        # elif pos == u"Pron.":
        case (u"Pron.", subc, _):
            if subc == None:
                return Pos.PPER.plain  # PPER'
            elif subc == u"Indefinitpronomen" or subc == u"Fragepronomen":
                return Pos.PINT.plain  # 'PINT'
            elif subc == u"Reflexivpronomen":
                return Pos.PREP.plain
        # elif pos == u"Satzkonverter"
        # Sentence converter which is unique to Coptic
        case (u"Satzkonverter", _, _):
            return Gram.C.plain
        # elif pos == u"Präfix":
        case (u"Präfix", _, orthstring) if orthstring is not None:
            if u"ⲧⲁ-" in orthstring:
                return Pos.PPOS.plain  # "PPOS"
            elif u"ⲧⲃⲁⲓ-" in orthstring:
                return Pos.Noun.plain  # "N"
            elif u"ⲧⲣⲉ-" in orthstring:
                return Pos.A.plain  # "A"
        # elif pos == u"None" or pos == u"?":
        case (None, _, _)| (u"?", _, _):
            if subc == u"None" or subc == 'None':
                return 'NULL'
            if subc == u"Qualitativ":
                return Pos.VSTAT.plain
        case (_, _, orthstring) if orthstring is not None and u"ϭⲁⲛⲛⲁⲥ" in orthstring:
            return "None"

        case _:
            return "?"

def pos_map(pos, subc, orthstring):
	"""
	:param pos: string
	:param subc: string
	:return: string
	"""
	pos = pos.replace('?', '')
	if pos == u"Subst." or pos == u"Adj." or pos == u"Nominalpräfix" or pos == u"Adjektivpräfix" \
			or pos == u"Kompositum":
		return 'N'
	elif u"Ausdruck der Nichtexistenz" in subc or u"Ausdruck des Nicht-Habens" in subc:
		return 'EXIST'
	elif pos == u"Adv.":
		return 'ADV'
	elif pos == u"Vb." or pos == u"unpersönlicher Ausdruck":
		if subc == u"Qualitativ":
			return 'VSTAT'
		elif subc == u"Suffixkonjugation":
			return 'VBD'
		elif subc == u"Imperativ":
			return 'VIMP'
		elif u"ⲟⲩⲛ-" in orthstring or u"ⲟⲩⲛⲧⲉ-" in orthstring:
			return "EXIST"
		else:
			return 'V'
	elif pos == u"Präp.":
		return 'PREP'
	elif pos == u"Zahlzeichen" or pos == u"Zahlwort" or pos == u"Präfix der Ordinalzahlen":
		return 'NUM'
	elif pos == u"Partikel" or pos == u"Interjektion" or pos == u"Partikel, enklitisch":
		return 'PTC'
	elif pos == u"Selbst. Pers. Pron." or pos == u"Suffixpronomen" or pos == u"Präfixpronomen (Präsens I)":
		return 'PPER'
	elif pos == u"Konj.":
		return 'CONJ'
	elif pos == u"Dem. Pron.":
		return "PDEM"
	elif pos == u"bestimmter Artikel" or pos == u"unbestimmter Artikel":
		return 'ART'
	elif pos == u"Possessivartikel" or pos == u"Possessivpräfix":
		return 'PPOS'
	elif pos == u"Poss. Pron.":
		return 'PPERO'
	elif pos == u"Interr. Pron.":
		return 'PINT'
	elif pos == u"Verbalpräfix":
		if subc == u"Imperativpräfix ⲁ-" or subc == u"Negierter Imperativ ⲙⲡⲣ-":
			return 'NEG'
		if subc == u"im negativen Bedingungssatz" or subc == u"Perfekt II ⲉⲛⲧⲁ-":
			return 'NONE'
		else:
			return 'A'
	elif pos == u"Pron.":
		if subc == "None":
			return 'PPER'
		elif subc == u"Indefinitpronomen" or subc == u"Fragepronomen":
			return 'PINT'
		elif subc == u"Reflexivpronomen":
			return 'PREP'
	elif pos == u"Satzkonverter":
		return 'C'
	elif pos == u"Präfix":
		if u"ⲧⲁ-" in orthstring:
			return "PPOS"
		elif u"ⲧⲃⲁⲓ-" in orthstring:
			return "N"
		elif u"ⲧⲣⲉ-" in orthstring:
			return "A"
	elif pos == u"None" or pos == u"?":
		if subc == u"None":
			return 'NULL'
		if subc == u"Qualitativ":
			return 'VSTAT'
	elif u"ϭⲁⲛⲛⲁⲥ" in orthstring:
		return "NULL"

	return "?"

@dataclasses.dataclass(frozen=True)
class Sense:
    # refs: list
    quotes: list
    is_ddglc: bool = False


@dataclasses.dataclass
class Entry:
    form: str
    # Grammatical information
    #  '{http://www.tei-c.org/ns/1.0}pos',
    #  '{http://www.tei-c.org/ns/1.0}gen',
    #  '{http://www.tei-c.org/ns/1.0}gram',
    #  '{http://www.tei-c.org/ns/1.0}note',
    #  '{http://www.tei-c.org/ns/1.0}number',
    #  '{http://www.tei-c.org/ns/1.0}subc'
    """.//tei:pos
    {'Interr. Pron.', '?', 'Partikel', 'Vb.', 'Suffixpronomen', 'Kompositum', 'Verbalpräfix', 'Adjektivpräfix', 'Zahlwort', 'Präfix', 'Subst.', 'Präp.', 'Präfix der Ordinalzahlen', 'Possessivartikel', 'unpersönlicher Ausdruck', 'Vb.?', 'Partikel, enklitisch', 'Vb. Hilfsverb.', 'Subst.?', 'bestimmter Artikel', 'unbestimmter Artikel', 'Possessivpräfix', 'Zahlzeichen', 'Präfixpronomen (Präsens I)', 'Konj.', 'Pron.', 'Adj.?', 'Satzkonverter', 'Nominalpräfix', 'Adv.', 'Adj.', 'Selbst. Pers. Pron.', 'Dem. Pron.', 'Poss. Pron.', 'Interjektion'}
    .//tei:gen
    {'m.', 'f.'}
    .//tei:number
    {'sg.', 'pl.'}
    .//tei:subc
    {'Temporalis ⲛⲧⲉⲣⲉ-', 'Qualitativ', 'Imperativ', 'Konjunktiv ⲛ(ⲧⲉ)-', 'Obsolete Relativform', 'Perfekt ⲁ-', 'Kausativer Infinitiv ⲧⲣⲉ-', 'Aorist ϣⲁ(ⲣⲉ)-', 'im negativen Bedingungssatz', 'Status pronominalis', 'Futur III neg. ⲛⲛⲉ-', 'Negationspartikel', 'Infinitiv', 'Kausativer Imperativ ⲙⲁⲣⲉ-', 'Umstandsatzkonverter', 'Aorist neg. ⲙⲉ(ⲣⲉ)-', 'Kompositverb (?)', 'Indefinitpronomen', 'Relativkonverter', 'Kompositverb', '3. Pers.', 'Völkername', 'Kopula', 'Perfekt neg. ⲙⲡⲉ-', 'Reflexivpronomen', 'Futur I / II ⲛⲁ-', 'Negativpräfix zur Bildung von Substantiven f.', 'Negiertes Futur im Bedingungssatz ⲁⲛⲛⲉ- (B)', 'Negierter Imperativ ⲙⲡⲣ-', 'Titel', 'Adjektivverb', 'Imperfekt ⲛⲉ(ⲣⲉ)-', 'Qualitativ?', 'Göttername', 'Fokalisierungskonverter', 'Konditionalis ⲉⲣϣⲁⲛ-', 'Imperativpräfix ⲁ-', 'Fragepartikel', 'Futur III ⲉ(ⲣⲉ)-', 'Frageadverb', 'Status nominalis', 'Perfekt II ⲉⲛⲧⲁ-', '1. Pers.', 'zur Bildung von Ortsangaben', 'Participium conjunctum', 'Fragepronomen', 'Ortsname', 'Name einer Sache oder Institution', 'Unpersönlich', '2. Pers. ', 'Genitivpartikel', 'Hilfsverb', 'Negativpräfix zur Bildung von Adjektiven', 'Suffixkonjugation', 'Kompletiv neg. ⲙⲡⲁ(ⲧⲉ)-'}
    .//tei:gram
    {'ⲥⲁⲃⲟⲗ', 'ⲛϩⲏⲧ', 'ⲛⲃⲟⲗ', 'ⲛⲧⲉⲛ', 'ϩⲁ-', 'ⲥⲁⲫⲁϩⲟⲩ', 'ⲛⲛⲁϩⲣⲛ-', 'ϩⲓⲣⲛ-', 'ⲉⲧⲟⲟⲧ⸗ ⲛ-', 'ⲉⲡϣⲱⲓ', 'ⲉϩⲟⲩⲛ-', 'ⲉϩⲣⲁⲓ ⲉ-', 'ⲥⲁⲙⲉⲛϩⲏ', 'ⲉⲃⲟⲗ ⲛ-', 'ⲉ- (+ Inf.)', 'ⲡⲣⲟⲥ', 'ⲛ-/ⲛⲁ⸗', 'ϩⲛ-', 'ⲉϩⲣⲛ-', 'ⲉⲧⲛ-', 'ϧⲁϫⲉⲛ-', 'ϩⲓⲣⲉⲛ-', 'ϣⲁ-', 'ⲉϩⲛ-', 'ⲉⲃⲟⲗ ϩⲛ-', 'ⲉⲣⲁⲧ⸗', 'ⲥⲁ-', 'ϩⲓⲡⲁϩⲟⲩ', 'ϩⲓ-', 'ⲉϫⲛ-', 'ⲉⲑⲏ', 'ⲉⲃⲟⲗ ⲉ-', 'ϫⲓⲛ-', 'ϩⲁϩⲧⲛ-', 'ⲛⲥⲁⲃⲟⲗ', 'ⲉϫⲉⲛ-', 'ϫⲉ', 'ⲉⲃⲟⲗ ϩⲓⲧⲛ-', 'ⲧⲟⲟⲧ⸗ ⲛⲥⲁ', 'ⲙⲛ-', 'ⲉϩⲟⲩⲛ', 'ⲛⲧⲉⲛ-', 'ⲉⲡϧⲣⲏⲓ', 'ⲟⲩⲃⲉ-', 'ⲉϧⲟⲩⲛ', 'ⲉϩⲣⲁⲓ ϩⲓϫⲛ-', 'ⲛⲥⲁ-', 'ⲉⲡⲁϩⲟⲩ', 'ⲉⲃⲟⲗ ⲛⲉⲙ- (B)', 'ⲙⲛⲛⲥⲁ-', 'ⲉ', 'ⲉⲧⲑⲏ', 'ⲛⲉⲙ-', 'ⲟⲩⲧⲉ-', 'ⲛⲥⲁ', 'ϩⲁⲑⲏ', 'ϩⲓⲃⲟⲗ', 'ⲉⲡⲱϣⲓ', 'ϩⲏⲧ⸗', 'ⲛ- (+ Inf.)', 'ⲛⲧⲛ-', 'ⲉϧⲣϩⲓ', 'ϩⲓϩⲏ', 'ⲉⲧⲟⲟⲧ⸗', 'ϩⲓϫⲛ-', 'ⲉⲣⲛ-', 'ϩⲓⲑⲏ', 'ⲉⲡϣⲏⲓ', 'ⲉⲧⲙⲏⲧⲉ', 'ⲉϩⲟⲩⲛ ⲉ-', 'ⲉϩⲣⲁⲓ', 'ϩⲓⲧⲛ-', 'ⲙⲡⲉ', 'ϩⲏⲧ', 'ⲛⲧⲟⲟⲧ⸗ ⲉ-', 'ϩⲓϫⲉⲛ-', 'ⲉ-', 'ⲉⲧⲃⲉ', 'ϩⲛ', '(ⲛ)ⲛⲁϩⲣⲛ-', 'ⲉⲃⲟⲗ', 'ⲛ-', 'ϩⲁⲣⲁⲧ-', 'ⲙⲡⲱⲣ', 'ⲉⲡⲉⲥⲏⲧ', 'ⲉϩⲣⲉⲛ-', 'ⲛⲁ⸗', 'ⲙⲙⲁⲩ', 'ⲉⲃⲟⲗ ϩⲓϫⲛ-', 'ϣⲟⲩϩⲏⲛⲉ', 'ϧⲉⲛ-', 'ⲉⲣⲁⲧ-', 'ⲙⲛ- ', 'ⲛⲁϩⲣⲛ-', 'ϩⲁϫⲛ-', 'ϩⲁⲃⲟⲗ-', 'ϣⲁ', 'ⲛ', 'ⲉϩⲟⲩⲛ (ⲉ-)', 'ϫⲓⲛ', 'ⲁⲃⲁⲗ', 'ⲙⲙⲟⲛ', 'ⲛ-, ⲉ-', 'ϩⲁ(ϩ)ⲧⲛ-', 'ⲙⲃⲟⲗ', 'ⲉϩⲣⲁⲓ ⲉϫⲛ-', 'ϩⲁⲣⲛ-', 'ⲥⲁⲃⲟⲗ ⲛ-'}"""
    pos: str
    gen: str
    number: str
    subc: str
    # gram: str
    # note: str
    senses: list

    @staticmethod
    def subc_map(subc, *args):
        return coptic_grammar_indicators.get(subc, None)


    @staticmethod
    def make_entry(xml_entry, form, ns, pos_tags, target_langs, max_senses):
        # TODO maybe use note also
        def gen_map(gen, *args):
            if gen == 'm.':
                return 'Masculine'
            elif gen == 'f.':
                return 'Feminine'
            else:
                return None

        def num_map(num, *args):
            if num == 'sg.':
                return 'Singular'
            elif num == 'pl.':
                return 'Plural'
            else:
                return None

        tags = {
            'pos' : new_pos_map,
            'gen': gen_map,
            'number': num_map,
            'subc' : lambda x, *args: None
        }
            #'subc', 'gram'], 'note']
        def get_grammatical_information():
            return {
                tag: (xml_entry.find(f".//tei:{tag}", ns).text if xml_entry.find(f".//tei:{tag}", ns) is not None else None)
                for tag in tags
            }

        grammar_info = get_grammatical_information()
        # Map POS
        # TODO: abhip map other grammatical information too
        # Map grammatical information
        for tag in grammar_info:
            grammar_info[tag] = tags[tag](grammar_info[tag], grammar_info['subc'], form)

        senses = list()
        ddglc_bibls = list()
        # TODO implement sense filtering heuristics
        for sense in xml_entry.iterfind(".//tei:sense", ns):
            # Filter quotes and refs by langs
            # sense_refs = [(ref.attrib['type'],  ref.text) for ref in sense.iterfind(".//tei:ref", ns) if 'type' in ref.attrib and ref.attrib['type'] in target_langs]
            def lang(quote):
                return quote.attrib['{' +ns['xml'] + '}lang']
            # Some cits can be examples rather than translations
            # cit type='example' vs cit type='translation'
            sense_quotes = [
                [lang(quote), quote.text] for cit in sense.iterfind('.//tei:cit[@type="translation"]', ns) \
                                for quote in cit.iterfind(".//tei:quote", ns) \
                if lang(quote) in target_langs and quote.text is not None]

            # DDGLC auditing
            bibls = [bibl.text for cit in sense.iterfind('.//tei:cit[@type="translation"]', ns) \
                                for bibl in cit.iterfind(".//tei:bibl", ns)]
            ddglc_bibls.append([True if bibl == 'DDGLC' else False for bibl in bibls])
            ddglc_counts = sum(1 for bibl in bibls if bibl == 'DDGLC')

            if len(sense_quotes) == 0:
                logger.warning(f"No quotes found for sense in form {form} with id {sense.attrib.get('{' + ns['xml'] + '}id', 'N/A')}")
            else:
                is_ddglc = ddglc_counts > 0
                if is_ddglc is True:
                    logging.info(f"Marking sense as DDGLC/Senses, with {ddglc_counts:02}/{len(sense_quotes):02}")
                senses.append(Sense(sense_quotes, is_ddglc))

        logger.info(f"{len(senses)} senses for form {form}")
        # Combine DDGLC entries for senses
        return Entry(form, grammar_info['pos'], grammar_info['gen'], grammar_info['number'], grammar_info['subc'], senses)

# Function which loads the XML and organizes the entries for augmentation
# TODO abhip: Avoid xpath (serial traversal) use (some form) of xml preprocessing for faster lookup
# TODO abhip: Avoid loading XML each time
languages_map = {
        'en': 'English',
        'de': 'German',
        'fr': 'French',
    }


def lang_for_code(code):
    if code not in languages_map:
        logger.error(f"Unknown language code: {code}")
    return languages_map.get(code, code)

class Lexicon:

    def __init__(self, lexicon_xml):
        self.lexicon_xml = lexicon_xml
        # TODO abhip check for default XML namespace handling
        self.ns = {
                    'tei': 'http://www.tei-c.org/ns/1.0',
                   'xml': 'http://www.w3.org/XML/1998/namespace'
                   }
        logger.info(f"Loading lexicon from {self.lexicon_xml}")
        self.tree = ET.parse(lexicon_xml)
        self.root = self.tree.getroot()
        # Set caches for lexicon
        self.form_entries_cache = dict()
        # Log a summary of the loaded dictionary
        self.log_summary()

        self.stats = dict()
        self.stats.setdefault('entries', list())


    def _log_entries(self, entries_found, entries_kept):
        self.stats['entries'].append({
            'found': entries_found,
            'kept': entries_kept
        })

    def log_summary(self):
        ns = self.ns
        xml_tree = self.tree
        n_superentries = len(xml_tree.findall('.//tei:superEntry', ns))
        n_entries = len(xml_tree.findall('.//tei:entry', ns))
        n_senses = len(xml_tree.findall('.//tei:sense', ns))
        pos_tags = sorted(set([x.text for x in xml_tree.findall('.//tei:entry//tei:pos', ns)]))
        logger.info(f"Number of superEntries: {n_superentries}")
        logger.info(f"Number of entries: {n_entries}")
        logger.info(f"Number of senses: {n_senses}")
        logger.info(f"Different parts of speech: {pos_tags}")
        # TODO summarize coverage

    def __del__(self):
        try:
            found_counts = [entry['found'] for entry in self.stats['entries']]
            logger.info("'found' Counts: %s", Counter(found_counts))

            kept_counts = [entry['kept'] for entry in self.stats['entries']]
            logger.info("'kept' Counts: %s", Counter(kept_counts))

            hit_rates = [entry['kept'] / entry['found'] if entry['found'] > 0 else 0.0 for entry in self.stats['entries']]
            avg_hit_rate = sum(hit_rates) / len(hit_rates) if len(hit_rates) > 0 else 0.0
            logger.info("Stats: Entries found vs kept per lookup: mean %s, min %s , max %s", avg_hit_rate, min(hit_rates), max(hit_rates))

        except Exception as e:
            logger.warning(f"Exception occurred in {self.__class__} __del__: {e}")


    def _convert_entry_to_dict(self, entry, form):
        entry_dict = dict()
        entry_dict['form'] = form
        senses = list()
        for sense in entry.iterfind(".//tei:sense", self.ns):
            sense_translations = list()
            # Get the ref, cit quotes

            for ref in sense.iterfind(".//tei:ref", self.ns):
                sense_translations.append({
                    "lang": ref.attrib['type'],
                    "text": ref.text
                })

            for cit in sense.iterfind(".//tei:cit", self.ns):
                for quote in cit.iterfind(".//tei:quote", self.ns):
                    sense_translations.append({
                        "lang": quote.attrib["{" + self.ns['xml'] + "}lang"],
                        "text": quote.text
                    })

            senses.append(sense_translations)
        return senses

    # Note this form can also be a lemma, or non-lemma
    def _find_entries_for_form(self, form, dialect):
        if (form, dialect) in self.form_entries_cache:
            logger.debug(f"Cache hit for form: {form} and dialect: {dialect.value}")
            return self.form_entries_cache[(form, dialect)]

        # TODO abhip: Using form means you are not looking for lemma
        entries_for_form = self.tree.findall(f".//tei:entry[tei:form]//tei:orth[. = '{form}']....", self.ns)
        # TODO abhip: Find entries for the lemma
        # TODO FIX: Filtering for dialect outside the XPath
        n_found = len(entries_for_form)
        logger.debug(f"Found entries for form: {len(entries_for_form)}")
        entries_for_form = [entry for entry in entries_for_form if entry.find(f".//tei:usg[. = '{dialect.value}']", self.ns) is not None]
        n_kept = len(entries_for_form)
        logger.debug(f"Found entries for form: {len(entries_for_form)} and dialect {dialect.value}")
        self.form_entries_cache[(form, dialect)] = entries_for_form
        logger.debug(f"Cached {len(entries_for_form)} entries for form: {form} and dialect: {dialect.value}")
        self._log_entries(n_found, n_kept)
        # TODO Super entries for the entries
        return entries_for_form


    def find_super_entries_for_entry(self, form, dialect):
        super_entries_for_form = self.tree.findall(f".//tei:orth[. = '{form}']......", self.ns)
        # TODO FIX: Filtering for dialect outside the XPath
        logger.debug(f"Found super entries for form: {len(super_entries_for_form)}")
        super_entries_for_form = [super_entry for super_entry in super_entries_for_form if super_entry.find(f".//tei:usg[. = '{dialect.value}']", self.ns) is not None]
        logger.debug(f"Found super entries for form: {len(super_entries_for_form)} and dialect {dialect.value}")
        return super_entries_for_form




        # Usages from Utility
        # Collocations from Utility
        return entries, super_entries

    # def make_lexicogrammatical_prompt(self, form, dialect,
    #                                   entry_langs= DictionaryLangs.values(),
    #                                   pos_tags = sorted([pos.plain for pos in Pos]),
    #                                   grammatical_info=True,
    #                                   max_entries=5,
    #                                   max_senses_per_entry=2):
    #     if entry_langs is None:
    #         entry_langs = DictionaryLangs.values()
    #     logger.info("Making lexicogrammatical info for form: {form} and dialect: {dialect}")
    #     logger.info(f"With knobs: grammatical_info={grammatical_info}, max_entries={max_entries}, max_senses_per_entry={max_senses_per_entry}")
    #     entries = self.find_entries_for_form(form, dialect)
    #     # Filter entries by pos
    #     # Filter quotes and refs by langs
    #     entries = [Entry(entry, form, self.ns) for entry in entries]
    #     return entries, super_entries

    # TDOD
    # def make_prompt_for_entry(entry):
    #     # TODO abhip: Use super entries too
    #     for i, sense in enumerate(entry.senses[:max_senses_per_entry]):
    #         # Include only passed langs
    #         sense_text= "\n".join([f"- In {ref_type}, {form} means {ref_text}" for ref_type, ref_text in sense.refs if ref_type in entry_langs])
    #         quote_text = "\n".join([f"- In {quote_lang}, {form} means {quote_text}" for quote_lang, quote_text in sense.quotes if quote_lang in entry_langs]) \
    #
    #     # TODO handle none values for grammar fields
    #     # Adding specific grammar information to the prompt
    #
    #     grammar_string = ", ".join([f for f in [entry.pos, entry.gen, entry.number, entry.subc] if f is not None])
    #     return (
    #         f"{entry.form} is {grammar_string}.\n"
    #         f"{sense_text}"
    #         f"{quote_text}"
    #     )
    #     # Filter entries by pos

    @staticmethod
    def compress_senses(uncombined_senses):
        logging.info(f"Attempting to compress {len(uncombined_senses)} senses")

        def compress_without_paranthetical():
            compressed_senses = list()
            unparanthetical_quotes = set()
            for sense in uncombined_senses:
                lang, quote = sense.quotes[0]
                unparanthetical = quote.split("(")[0].strip()
                if ((lang, unparanthetical) not in unparanthetical_quotes) or (not sense.is_ddglc):
                    compressed_senses.append(sense)
                    unparanthetical_quotes.add((lang, unparanthetical))
                else:
                    logger.info(f"Compressing sense {sense.is_ddglc} with quote:{quote}")
            return compressed_senses

        compressed_senses = compress_without_paranthetical()
        if len(compressed_senses) < len(uncombined_senses):
            logging.info(
                f"compressed to {len(compressed_senses):02} from {len(uncombined_senses):02} senses with unparanthetical filtering")

        def before_comma():
            first_term_senses = list()
            first_terms = set()

            # Note: This uses 'compressed_senses'n from outside sc
            for sense in compressed_senses:
                lang, quote = sense.quotes[0]
                first_term = quote.split(",")[0].strip()
                if (lang, first_term) not in first_terms or (not sense.is_ddglc):
                    first_term_senses.append(sense)
                    first_terms.add((lang, first_term))
                else:
                    logger.info(f"Compressing sense {sense.is_ddglc} with quote:{quote}")
            return first_term_senses

        compressed_senses = before_comma()
        if len(compressed_senses) < len(uncombined_senses):
            logging.info(
                f"compressed to {len(compressed_senses):02} from {len(uncombined_senses):02} senses with 'before comma' filtering")
        else:
            logging.info("No compression of senses.")

        return compressed_senses

    def _get_data_for_sentence(self, tokens, dialect, entry_langs, pos_tags_setting,
                               dedup_senses, grammatical_info, max_entries, max_senses_per_entry):
        # TODO token set optimization
        forms, lemmas, pos, msegs = tokens
        sentence_data = list()
        match pos_tags_setting:
            case "all":
                pos_tags = sorted([pos.plain for pos in Pos])
            case _:
                raise ValueError("Bad pos tags setting")
        for form, lemma, mseg in zip(forms, lemmas, msegs):
            # TODO implement caching of entry lookups

            entries = self._find_entries_for_form(form, dialect)
            #Collect entries for lemma
            #print(form, lemma)

            if lemma != form:
                logger.info(f"Lemma {lemma} is not the same as the {form}")
                entries_for_lemma = self._find_entries_for_form(lemma, dialect)
                entries = entries + entries_for_lemma

            # TODO make this a fall back only if no entries found for form/lemma
            if mseg is not None:
                segs = mseg.split('-')
                logger.info(f"Using morphological segments to find entries {segs}")
                for seg in segs:
                    seg_entries = self._find_entries_for_form(seg, dialect)
                    entries.extend(seg_entries)
            # TODO make filter by pos happen here
            # knob: Filter entries by pos
            entries = [Entry.make_entry(entry, form, self.ns, pos_tags, entry_langs, max_senses_per_entry) for entry in entries]
            entries = filter(lambda entry: entry.pos in pos_tags, entries)
            # Make list since we are doing assignments
            entries = list(entries)
            # Knob: compress senses
            if dedup_senses:
                for entry in entries:
                    entry.senses = self.compress_senses(entry.senses)
            # Knob: limit senses per entry
            for entry in entries:
                entry.senses = entry.senses[:max_senses_per_entry]

            # Knob: limit entries per form
            entries = entries[:max_entries]
            # TODO: FIXME super entries are unsued
            super_entries = self.find_super_entries_for_entry(form, dialect)
            # logger.info(f"{len(entries)} entries kept after POS filtering for form: {form}")
            def total_senses(entry_list):
                return sum([len(entry.senses) for entry in entry_list])

            sentence_data.append(entries)
            logger.debug(f"Total {total_senses(entries):02} senses for token {form}")
            logger.info(f"Total {len(entries):02} entries for token {form} after POS filtering")
        return sentence_data

    @staticmethod
    def _construct_prompt_text(lexicon_info, tokens):
        def make_info_for_entry(token, entry):

            entry_text =  fr"Dictionary entry {entry.pos} {entry.form}" \
                          + fr"{', ' + entry.gen if entry.gen else ''}" \
                          + fr"{', ' + entry.number if entry.number else ''}" \
                          + fr"{', ' + entry.subc if entry.subc else ''}" \
                          + fr" has {len(entry.senses)} senses.\n"

            senses_text = ""
            for i, sense in enumerate(entry.senses):
                # Encode quote [1]
                def escape(text):
                    return text
                    # if text is None:
                    #     logger.warning("None text found in quote value.")
                    #     return None
                    # return codecs.encode(text, "unicode-escape").decode('utf-8')

                senses_text += fr"Sense {i+1}:\n" + r"\n".join(
                    [fr"- In {lang_for_code(quote[0])}, {token} means {escape(quote[1])}" for quote in sense.quotes]
                ) + r"\n"

            return entry_text + senses_text

        source_language = "Coptic"
        # Adapted from https://aclanthology.org/2025.acl-long.429/
        prompt_text = (
            fr"For the translation task, you are given dictionary entries for {source_language}."
            fr" Some words can be polysemous and there might be multiple entries."
            fr" Each entry can contain multiple senses with translations in {[lang_for_code(code) for code in DictionaryLangs.values()]}."
            fr" In such a case, please choose the most appropriate one."
            fr" Note that for some words, they might be derived from a more basic form, some entries will be for such lemma."
            fr" Here are the entries for collected for individual words in the sentence:\n"
        )

        n_entries = 0
        n_tokens = 0
        for info, token in zip(lexicon_info, tokens):
            if len(info) > 0:
                n_tokens += 1

            for entry in info:
                prompt_text += make_info_for_entry(token, entry)
                n_entries += 1
        logger.info(f"Used: {n_entries} entries for {n_tokens} tokens")

        if n_entries < 1:
            logger.warning(f"No entries found for any token in the sentence {tokens}!")
            return ""
        return prompt_text

    def get_lexicon_prompt_text(self, forms, lemmas, uposes, msegs, retrieval_config, dialect=CopticDialect.S,
                                ):
        logging.info("Using lexicon retrieval config: %s", retrieval_config)

        filter_content_upos = True
        # TODO refactor this to avoid code duplication
        if filter_content_upos is True:
            content_upos = {'NOUN', 'VERB', 'PROPN', 'ADV', 'ADJ'}
            tokens = (
                [form for form, upos in zip(forms, uposes) if upos in content_upos],
                [lemma for lemma, upos in zip(lemmas, uposes) if upos in content_upos],
                [upos for upos in uposes if upos in content_upos],
                [mseg for mseg, upos in zip(msegs, uposes) if upos in content_upos]
            )
        else:
            tokens = (
                forms,
                lemmas,
                uposes,
                msegs,
            )

        # Knobs for lexicon data extraction
        lexicon_info = self._get_data_for_sentence(tokens, dialect,
                                                   retrieval_config["entry_langs"],
                                                   retrieval_config["pos_tags"],#[Pos.Noun.plain, Pos.Verb.plain, Pos.PREP.plain, Pos.Adverb.plain, Pos.A.plain],
                                                   retrieval_config["dedup_senses"],
                                                   retrieval_config["grammatical_info"],
                                                   retrieval_config["max_entries"],
                                                   retrieval_config["max_senses_per_entry"]
        )
        return self._construct_prompt_text(lexicon_info, forms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process a Coptic dictionary XML file.')
    parser.add_argument('--lexicon', type=str, help='Path to the input XML file')
    # CLI argument for the enum CopticDialect
    parser.add_argument('--dialect', type=str, choices=[dialect.name for dialect in CopticDialect]
                        , help='Coptic dialect of input', default=CopticDialect.S.name)
    parser.add_argument('--conllu', type=str, help='Path to the output XML file')
    args = parser.parse_args()

    dialect = CopticDialect(args.dialect)
    lexicon = Lexicon(args.lexicon)
    conllu = pyconll.load_from_file(args.conllu)

    logger.info(f"Processing {args.conllu} with lexicon {args.lexicon} for {dialect.value}")
    # Todo check why there are two different entries for the same form
    config = configparser.ConfigParser()
    config.read("best.components.ini")
    for conllu_sentence in conllu[:1]:
        def get_mseg(token):
            if "MSeg" in token.misc:
                return list(token.misc["MSeg"])[0]
            else:
                return None
        print(lexicon.get_lexicon_prompt_text(
            [token.form for token in conllu_sentence],
            [token.lemma for token in conllu_sentence],
            [token.upos for token in conllu_sentence],
            [get_mseg(token) for token in conllu_sentence],
            dict(config.items("lexicon"))
        ))

    print("Entries stats", lexicon.stats["entries"])
