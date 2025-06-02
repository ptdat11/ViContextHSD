import pandas as pd
import numpy as np
import json
import torch
import unicodedata
import os
import re
from tqdm import tqdm
from typing import Callable, Optional, Any, Sequence
from utils.logger import Logger
from itertools import compress
from together import Together

# from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
# if "tokenizer" not in globals():
#     tokenizer = None
# if "model" not in globals():
#     model = None
# if "dep_parse" not in globals():
#     dep_parse = None

from py_vncorenlp import VnCoreNLP
if "vncorenlp" not in globals():
    vncorenlp = None

def init_word_segmenter():
    # global tokenizer, model, dep_parse
    # if not tokenizer:
    #     tokenizer = AutoTokenizer.from_pretrained("NlpHUST/vi-word-segmentation")
    # if not model:
    #     model = AutoModelForTokenClassification.from_pretrained("NlpHUST/vi-word-segmentation")
    # if not dep_parse:
    #     dep_parse = pipeline("token-classification", model=model, tokenizer=tokenizer, device_map="auto")
    global vncorenlp
    if not vncorenlp:
        cwd = os.getcwd()
        vncorenlp = VnCoreNLP(annotators=["wseg"], save_dir=os.environ['VNCORENLP'])
        os.chdir(cwd)

# def ner_result_to_words(ner_result: list[dict]):
#     seq_len = len(ner_result)
#     return [
#         (word := (f"{anno["word"]} {ner_result[i+1]["word"]}" if i+1 < seq_len and ner_result[i+1]["word"] == "I" 
#         else f"{anno["word"]}{ner_result[i+1]["word"].replace("##", "")}" if i+1 < seq_len and ner_result[i+1]["word"].startswith("##") 
#         else anno["word"]).lower().strip())
#         for i, anno in enumerate(ner_result)
#         if anno["entity"] != "I"
#     ]

newline_re = re.compile(r'(?<![.!?])\s*\n\s*')
def normalize(text: str):
    text = text.lower()
    text = newline_re.sub(". ", text)
    text = unicodedata.normalize("NFKC", text)
    return text

# @torch.inference_mode()
def word_segment(text: str | list[str]) -> list[list[str]]:
    init_word_segmenter()
    global vncorenlp
    if isinstance(text, str):
        text = [text]

    text = [
        [
            (word if len(word) == 1 else word.replace("_", " ")).strip()
            for sent in vncorenlp.word_segment(normalize(t))
            for word in sent.split()
        ]
        for t in text
    ]
    return text

#     ner_results = dep_parse(text)
#     ner_results = [ner_result_to_words(res) for res in ner_results]
#     return ner_results


class LlamaVisionAugmenter:
    SYSTEM_PROMPT = """INSTRUCTION:
You are an expert in language data augmentation in hate and offensive speech detection domain. You will be provided with a post from Facebook. The post will be comprising of a Post Caption, a Post Image, its one Comment and a Reply to the comment (in Vietnamese). Your task is to REPHRASE each of the Post Caption, Comment and Reply so that they are different from the original wording.
- Rephrased version MUST carry the same meaning and intention of the commenter, especially the same Offensive or Hateful level.
- You are ALLOWED to generate hateful or offensive content if they are originally hate speech as the purpose of the task is to fight hate speech.
- You are ALLOWED to and MUST generate vulgar slangs if they appears in original text. Example: "Hay vl" -> "Đm hay vãi", because speech contains "vl", the new text must contain a vulgar slang ("đm" or others).
- You should corporate many provided sources of information to capture the their meaning.

RECOMMENDED AUGMENTATION METHODS:
- Contextual paraphrasing: paraphrase sentences in the text while considering the surrounding context. this ensures that
the paraphrased sentences maintain the same meaning within the given context.
- Synonym Replacement: Replace certain words in the text with their synonyms while keeping the sentence structure
intact.
- Character-level augmentation: modify individual characters within words, such as changing vowels or consonants, to
generate diverse textual variations.
- Sentence splitting/merging: split long sentences into shorter ones or merge short sentences into longer ones to create new
sentence structures.
- Synonym replacement: replace words in the text with their synonyms. this can create alternative phrasing while
maintaining the overall meaning.
- Part-of-speech tagging: modify the part-of-speech tags of words in the text to create new grammatical arrangements and
sentence structures.
- Named entity replacement: identify named entities in the text (e.g., names of people, organizations) and replace them
with similar entities to generate new variations.
- Social media noise: add noise to the text to mimic social media data, e.g. mispelling, code-mixing, abbreviation...

OUTPUT FORMAT:
{
    "caption": <rephrased caption>,
    "comment": <rephrased comment>,
    "reply": <rephrased reply>
}

- Rephrased sentences must be Vietnamese
- You MUST ONLY response with the JSON
- If the user doesn't provide Reply, DON'T include "reply" key-value in output JSON"""
    def __init__(
        self, 
        temperature: float = 1, 
        api_key: str = "tgp_v1_eQ2K-4iUHRkzyrqSvgj6nKyUbxx6QP3xfLwIl-A2C2Q"
    ):
        self.temperature = temperature
        self.client = Together(api_key=api_key)

    def llm_augment(self, row: pd.Series):
        caption = row.caption
        reply = row.comment if row.parent else None
        comment = row.parent_text if row.parent else row.comment
        image_url = f"https://res.cloudinary.com/ptdat11/image/upload/{row.image}"

        user_prompt = f"""Post Caption: ```{caption}```
Comment: ```{comment}```"""
        if reply:
            user_prompt += f"\nReply: ```{reply}```"

        response = self.client.chat.completions.create(
            model="meta-llama/Llama-Vision-Free",
            messages=[
                {"role": "system", "content": LlamaVisionAugmenter.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                    ]
                }
            ]
        )
        print(response.choices[0].message.content)
        return json.loads(response.choices[0].message.content)
    
    def __call__(self, row: pd.Series):
        response = self.llm_augment(row)
        caption = response["caption"]
        comment = response["reply"] if "reply" in response and "parent_text" in row else response["comment"]
        parent_text = response["comment"] if "reply" in response else None

        _row = row.copy()
        _row.caption = caption
        _row.comment = comment
        if parent_text and "parent_text" in row.index:
            _row.parent_text = parent_text
        
        return _row


class SynonymReplacer:
    def __init__(self, wordnet_json: str, rate: float = 0.1, stopword_txt: str | None = None):
        self.wordnet_json = wordnet_json
        self.stopword_txt = stopword_txt
        self.rate = rate

        if stopword_txt:
            with open(stopword_txt) as f:
                self.stopwords = set(f.read().split("\n"))
        else: self.stopwords = []

        with open(wordnet_json, "r") as f:
            self.wordnet = json.load(f)

    def _replace_word_with_synonym(self, word: str):
        synonyms = self.wordnet.get(word, [])
        choice = np.random.choice(synonyms) if synonyms else word
        return choice
    
    def _replace_text(self, words: list[str]):
        ret_words = words.copy()
        seq_len = len(words)
        word_idx = np.arange(seq_len)
        total_replace = int(np.ceil(self.rate * seq_len))
        is_stopword = np.isin(words, self.stopwords)
        replace_idx = np.random.choice(word_idx[~is_stopword], size=total_replace, replace=False)
        for i in replace_idx:
            ret_words[i] = self._replace_word_with_synonym(ret_words[i])
        return ret_words

    def __call__(self, seqs: list[list[str]]):
        seqs = [
            self._replace_text(words)
            for words in seqs
        ]
        return seqs
    
class RandomWordRemover:
    def __init__(self, p: float = 0.1):
        self.p = p

    def _random_remove(self, words: list[str]):
        seq_len = len(words)
        if seq_len == 1:
            return words
        
        rand_nums = np.random.rand(seq_len)
        keep_idx = rand_nums > self.p
        if np.all(~keep_idx):
            dont_del_idx = np.random.choice(seq_len)
            keep_idx[dont_del_idx] = True
        
        words = list(compress(words, keep_idx))
        return words

    def __call__(self, seqs: list[list[str]]):
        seqs = [
            self._random_remove(words)
            for words in seqs
        ]
        return seqs
    

class TextProcessChain:
    def __init__(
        self,
        *steps
    ):
        self.steps = list(steps)
    
    def __call__(self, text: str | list[str]):
        seqs = word_segment(text)
        for step in self.steps:
            seqs = step(seqs)
        seqs = [
            " ".join(seq)
            for seq in seqs
        ]
        if isinstance(text, str):
            seqs = seqs[0]
        return seqs

class Augmenter:
    def __init__(
        self,
        label_col: str,
        col_apply_funcs: dict[str | tuple[str], Callable[[Any], Any]],
    ):
        self.label_col = label_col
        self.col_apply_funcs = col_apply_funcs
    
    @staticmethod
    def create_augment_distribution(label_counts: Sequence):
        label_counts = np.array(label_counts)
        p = label_counts / label_counts.sum()
        dist = 1 / np.sum(1/p) / p
        return dist
    
    def augment(self, data: pd.DataFrame, n: int) -> pd.DataFrame:
        label_counts = data[self.label_col].value_counts()
        augmented_idx = []
        for i in range(n):
            dist = self.create_augment_distribution(label_counts)
            sample_p = data[self.label_col].map(dist / data[self.label_col].value_counts())

            idx = np.random.choice(data.index, p=sample_p)
            augmented_idx.append(idx)
            label_counts[data.loc[idx, self.label_col]] += 1

        augmented_data = data.loc[augmented_idx].copy()
        for col, mapper in self.col_apply_funcs.items():
            if isinstance(col, tuple):
                col = list(col)
            augmented_data[col] = augmented_data[col].apply(mapper, axis=1)
        return augmented_data