"""
中文：英汉词典——基于开源 ECDICT 词库的离线查词。

Offline English-Chinese dictionary.

Backed by [ECDICT](https://github.com/skywind3000/ECDICT), an MIT-licensed
dataset of about 770,000 entries with phonetics, part-of-speech ratios, exam
tags and corpus frequency ranks.

The dictionary is deliberately not the translation model. Looking a word up
should be instant, work with no network and no GPU, and give the same answer
every time — a model call satisfies none of those. It also answers a different
question: a dictionary lists a word's senses, while a translator commits to one
reading of one sentence.
"""

from local_translator.dictionary.store import DictionaryStore, get_dictionary

__all__ = ["DictionaryStore", "get_dictionary"]
