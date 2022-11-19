import pandas as pd
import numpy as np
import transformers
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import random
from scipy.spatial.distance import cdist
import string
import re
import json
import joblib
import spacy
import random, math
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
import os
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk
import os
import sentence_transformers

print("sentence_transformers",sentence_transformers.__version__)
dir_path = os.path.dirname(os.path.realpath(__file__))

nltk.download('punkt')
from similarity.normalized_levenshtein import NormalizedLevenshtein

import warnings
warnings.filterwarnings("ignore")

"""# ParaQD Utils"""

def get_score(model, sentence_1, sentence_2):
    emb_1 = model.encode(sentence_1,  convert_to_tensor = True)
    emb_2 = model.encode(sentence_2,  convert_to_tensor = True)
    cosine_scores = util.pytorch_cos_sim(emb_1, emb_2)
    return cosine_scores

PARAQD_PATH = os.path.join(dir_path, '../models/ParaQD_Aqua') # Change the path
print("PARAQD_PATH",PARAQD_PATH)
paraqd = SentenceTransformer(PARAQD_PATH, device='cpu')

"""# Corruption Utils"""

def untokenize(words):
    """
    ref: https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py#L28
    """
    text = " ".join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace(". . .", "...")
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r" ([.,:;?!%]+)$", r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace("can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

class Corrupter(object):
    """Data augmentation operator.
    Support both span and attribute level augmentation operators.
    """
    def __init__(self, idf_fn='aqua_idf_dict', unit_file="UnitList.json"):
        # self.index = json.load(open(index_fn))
        self.nlp = spacy.load("en_core_web_sm")
        if idf_fn is not None:
            self.idf_dict = joblib.load(idf_fn)
        else:
            self.idf_dict = None
        with open(unit_file, "r") as f:
            self.units = set(json.load(f))
        self.nl = NormalizedLevenshtein()

    def is_part_of(self, token, wordlist):
        token1 = str(token)
        for word in wordlist:
            if token1 in str(word) or str(word) in token1:
                return True
        return False
        
    def check_ner_pos(self, token, doc):
        ne = doc.ents
        pos = set(["VERB", "ADJ", "NUM", "PROPN", "ADV", "PUNCT"])
        if self.is_part_of(token, ne): return True
        if type(token) == str: return False # Only possible if the token is a mask
        if token.pos_ in pos: return True
        return False

    def isValid(self, token, doc):
        """
        returns a boolean indicationg whether the token can be changed (not numeric or unit)
        """
        token1 = str(token)
        if re.search("\d+", token1) or token1 in self.units:
            return False
        if re.search(r"\[.+\]", token1) is not None: return False
        if self.check_ner_pos(token, doc): return False
        return True


    def mask(self, tokens, labels, doc):
        idx = self.sample_position(tokens, labels, doc)
        if idx < 0: return tokens, labels
        # print(tokens[idx])
        if type(tokens[idx]) != str:
            tokens[idx] = f"[{tokens[idx].pos_}]"
        return tokens, labels
        

    def augment(self, tokens, labels, doc, op='del', args=[]):
        tfidf = 'tfidf' in op
        if 'del' in op:
            if 'token_del' in op:
                pos1 = self.sample_position(tokens, labels, doc, tfidf)
                if pos1 < 0:
                    return tokens, labels
                new_tokens = tokens[:pos1] + tokens[pos1+1:]
                new_labels = labels[:pos1] + labels[pos1+1:]

        # elif 'repl' in op:
        #     pos1 = self.sample_position(tokens, labels, doc, tfidf)
        #     if pos1 < 0:
        #         return tokens, labels
        #     ins_token = self.sample_token(tokens[pos1])
        #     new_tokens = tokens[:pos1] + [ins_token] + tokens[pos1+1:]
        #     new_labels = labels[:pos1] + ['O'] + labels[pos1+1:]

        elif 'shuffle' in op:
            max_len = args[0] if len(args) > 0 else 4
            span_len = random.randint(2, max_len)
            pos1, pos2 = self.sample_span(tokens, labels, doc, span_len=span_len)
            if pos1 < 0:
                return tokens, labels
            sub_arr = tokens[pos1:pos2+1]
            random.shuffle(sub_arr)
            new_tokens = tokens[:pos1] + sub_arr + tokens[pos2+1:]
            new_labels = tokens[:pos1] + ['O'] * (pos2 - pos1 + 1) + labels[pos2+1:]

        elif 'mask' in op:
            new_tokens, new_labels = self.mask(tokens, labels, doc)

        else:
            new_tokens, new_labels = tokens, labels

        return new_tokens, new_labels


    def sample_span(self, tokens, labels, doc, span_len=3):
        candidates = []
        for idx, token in enumerate(tokens):
            if idx + span_len - 1 >= len(tokens) - 4: continue # Preserving the last few words, so that the qn doesnt change
            text = ' '.join(list(map(lambda x: str(x), tokens[idx:idx+span_len])))
            if idx + span_len - 1 < len(tokens) and re.search("\d+", text) is None:
                candidates.append((idx, idx+span_len-1))
        if len(candidates) <= 0:
            return -1, -1
        return random.choice(candidates)


    def sample_position(self, tokens, labels, doc, tfidf=False):
        candidates = []
        for idx, token in enumerate(tokens):
            if idx >= len(tokens) - 4: continue # Preserving the last few tokens
            candidates.append(idx)

        if len(candidates) <= 0:
            return -1

        if tfidf:
            oov_th = math.log(1e8)
            weight = {}
            max_weight = 0.0
            for idx, token in enumerate(tokens):
                token = str(token).lower()
                if token not in self.idf_dict:
                    self.idf_dict[token] = oov_th
                if token not in weight:
                    weight[token] = 0.0
                weight[token] += self.idf_dict[token]
                max_weight = max(max_weight, weight[token])

            weights = []
            for idx in candidates:
                weights.append(max_weight - weight[str(tokens[idx]).lower()] + 1e-6)

            weights = np.array(weights) / sum(weights)
            
            pointer = 0
            choices = np.random.choice(candidates, 2*len(candidates), p=weights)
            idx = choices[pointer]
            while (pointer < len(candidates) and not self.isValid(tokens[idx], doc)):
                pointer += 1
                idx = choices[pointer]
            return idx

        else:
            pointer = 0
            choices = np.random.choice(candidates, 2*len(candidates))
            idx = choices[pointer]
            while (pointer < len(choices)-1 and not self.isValid(tokens[idx], doc)):
                pointer += 1
                idx = choices[pointer]
            if self.isValid(tokens[idx], doc): return idx
            else: return -1

    def sample_token(self, token, max_candidates=10):
        """ Randomly sample a token's similar token stored in the index
        Args:
            token (str): the input token
            same_length (bool, optional): whether the return token should have the same
                length in BERT
            max_candidates (int, optional): the maximal number of candidates
                to be sampled
        Returns:
            str: the sampled token (unchanged if the input is not in index)
        """
        token = str(token).lower()

        candidates = []
        syns = wordnet.synsets(token)
        for syn in syns:
            for lem in syn.lemmas():
                w = lem.name().lower()
                if w != token and w not in candidates and '_' not in w:
                    candidates.append(w)

        # print(candidates)
        if len(candidates) <= 0:
            return token
        else:
            return random.choice(candidates)

    def corrupt(self, text, op='all', args=[]):
        """ Performs data augmentation on a classification example.
        Similar to augment(tokens, labels) but works for sentences
        or sentence-pairs.
        Args:
            text (str): the input sentence
            op (str, optional): a string encoding of the operator to be applied
            args (list, optional): the augmentation parameters (length, etc.)
        Returns:
            str: the augmented sentence
        """
        tokens = []
        doc = self.nlp(text)
        for token in doc:
            tokens.append(token)

        # avoid the special tokens
        labels = []
        for token in tokens:
            labels.append('O')

        N = len(tokens)
        num_augs = 0

        augs = np.random.choice(["del", "shuffle", "mask"], size=np.random.randint(2,4), replace=False)

        if "del" in augs:
            to_change = max(1, N // np.random.randint(4, 7))
            for _ in range(to_change):
                tokens, labels = self.augment(tokens, labels, doc, op='token_del_tfidf')
                # print(tokens, "DEL")
            num_augs += 1

        # N = len(tokens)
        # if np.random.rand() < 0.5:
        #     to_change = N // np.random.randint(4, 8)
        #     for _ in range(to_change):
        #         tokens, labels = self.augment(tokens, labels, doc, op='token_repl_tfidf')
        #         print(tokens, "REPL")
        #     num_augs += 1

        if "shuffle" in augs:
            to_change = max(1, N // np.random.randint(4, 7))
            for _ in range(to_change):
                tokens, labels = self.augment(tokens, labels, doc, op='shuffle')
                # print(tokens, "SHUFFLE")
            num_augs += 1

        if "mask" in augs:
            to_change = max(1, N // np.random.randint(4, 6))
            for _ in range(to_change):
                tokens, labels = self.augment(tokens, labels, doc, op='mask')
                # print(tokens, "MASK")

        tokens = list(map(lambda x: x.orth_ if type(x)!=str else x, tokens))
        results = untokenize(tokens)
        return results

    def corrupt_multiple(self, text, n=4):    
        corruptions = []
        for i in range(2*n):
            corruptions.append(self.corrupt(text))

        first = max(corruptions, key=lambda x: self.nl.distance(x, text))
        final_corruptions = [first]
        corruptions.remove(first)

        for i in range(n-1):
            next = max(corruptions, key=lambda x: self.nl.distance(x, text))
            final_corruptions.append(next)
            corruptions.remove(next)

        return final_corruptions

idf_fn = os.path.join(dir_path, '../files/aqua_idf_dict') # Change path
unit_file = os.path.join(dir_path, '../files/UnitList.json') # Change path

corrupter = Corrupter(idf_fn=idf_fn, unit_file=unit_file)

"""# Model Utils"""

BASE_PATH = "/content/drive/MyDrive/resources/models/" # Change path
MODEL_PATH = os.path.join(dir_path, '../models/bart_reconstruction_model_v3') # Change path
TOKENIZER_PATH = MODEL_PATH # Change path

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

prefix = ""

max_input_length = 256
max_target_length = 256

def paraphrase_outputs(test_samples, model, num_return=2):
    # model.eval()
    if type(test_samples) == str:
        test_samples = prefix + test_samples
    else:
        for i in range(len(test_samples)):
            test_samples[i] = prefix + test_samples[i]

    inputs = tokenizer(
        test_samples,
        truncation=True,
        padding="max_length",
        max_length=max_input_length,
        return_tensors="pt")
    
    input_ids = inputs.input_ids.to(model.device)
    # print(input_ids)
    # print(tokenizer.batch_decode(input_ids, skip_special_tokens=True))
    attention_mask = inputs.attention_mask.to(model.device)
    outputs = model.generate(input_ids, attention_mask=attention_mask, do_sample=True, top_p=0.95, top_k=50, max_length=max_target_length,  \
                             num_return_sequences=num_return) #, min_length=50)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_str

"""# Diversity"""

def diversity(inp, can):
    i_split = sent_tokenize(inp)
    c_split = sent_tokenize(can)
    smoothie = SmoothingFunction().method4
    ref, hyp = [], []
    if len(i_split)!=len(c_split):
        ref = [word_tokenize(inp)]
        hyp = word_tokenize(can)
        return 1-sentence_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    for i in i_split:
        tokens = [word_tokenize(i)]
        ref.append(tokens)
    for c in c_split:
        tokens = word_tokenize(c)
        hyp.append(tokens)
    return 1-corpus_bleu(ref, hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

"""# Postprocessing Utils"""

def read_file(file_path):
    with open(file_path, "r") as f:
        l = f.read().splitlines()
    return l

male_names = read_file(os.path.join(dir_path, '../files/male.txt'))
female_names = read_file(os.path.join(dir_path, '../files/female.txt'))
places = read_file(os.path.join(dir_path, '../files/places.txt'))

class PostProcessor:
    def __init__(self, male_names, female_names, places):
        self.nlp = spacy.load("en_core_web_sm")
        self.mn = male_names
        self.fn = female_names
        self.places = places

    def sample_element(self, ent, wordlist):
        n = len(wordlist)
        idx = np.random.randint(n)
        while wordlist[idx].lower() == ent.lower():
            idx = np.random.randint(n)
        return wordlist[idx]

    def find_replacement(self, ent, wordlist):
        words = ent.split()
        for word in wordlist:
            if word.lower() == words[0].lower():
                # words[0] = sample_element(ent, wordlist)
                # rep = " ".join(words)
                # return rep
                return self.sample_element(words[0], wordlist)
        return None

    def replace_entity(self, ent, type_, replacement):
        ent = str(ent); type_ = str(type_)
        if ent in replacement:
            return ent, replacement[ent]
        else:
            if type_ == "PERSON":
                rep = self.find_replacement(ent, male_names)
                if rep is None: rep = self.find_replacement(ent, female_names)
                ent = ent.split()[0]
                if rep is None: rep = ent
                replacement[ent] = rep
            elif type_ == "GPE":
                rep = self.find_replacement(ent, places)
                if rep is None: rep = ent
                replacement[ent] = rep
            else: 
                rep = ent
            return ent, rep

    def process(self, sentence):
        replacements = {}
        sent = sentence

        doc = self.nlp(sentence)
        for ent in doc.ents:
            ent, rep = self.replace_entity(ent.text, ent.label_, replacements)
            sent = sent.replace(ent, rep)

        return sent

post_processor = PostProcessor(male_names, female_names, places)


class Paraphraser:
    def __init__(self, reconstruction_model, selection_model, corrupter, post_processor):
        self.rec_model = reconstruction_model
        self.paraqd = selection_model
        self.corrupter = corrupter
        self.post_processor = post_processor
        self.string = ""

    def generate_corruptions(self, question, n=3, display=False):
        corruptions = self.corrupter.corrupt_multiple(question, n=n)
        string = "\n".join(corruptions)
        string = f"The corruptions are:\n{string}"
        if display: print(string)
        self.string += string
        return corruptions

    def generate_outputs(self, corruptions, num_return=2, display=False):
        outputs = paraphrase_outputs(corruptions, self.rec_model, num_return)
        return outputs

    def get_scores(self, original, outputs, weights, display=False):
        """
        returns the sorted unique scores 
        """
        sem_scores = get_score(self.paraqd, original, outputs)[0]
        scores = []
        for i in range(len(outputs)):
            sem_sc = sem_scores[i].item()
            div_sc = diversity(original, outputs[i])
            score = weights[0]*sem_sc + weights[1]*div_sc
            scores.append((i, sem_sc, div_sc, score))
        string = "\n".join(list(map(lambda x: " ".join(x), zip(outputs, map(lambda x: f"{x[1]:.2f} {x[2]:.2f} {x[3]:.4f}", scores)))))
        string = f"\nThe paraphrases and scores are:\n{string}\n"
        if display: print(string)
        self.string += string
        prev_scores = set([])
        unique_scores = []
        for score in scores:
            if score[-1] in prev_scores: continue
            prev_scores.add(score[-1])
            unique_scores.append(score)
        scores = sorted(unique_scores, key=lambda x: x[-1], reverse=True)
        return scores

    def paraphrase(self, question, top_n=2, n=4, num_return=2, weights=None, display=False, gradio=False):
        self.string = ""
        corruptions = self.generate_corruptions(question, n, display)
        outputs = self.generate_outputs(corruptions, num_return, display)
        scores = self.get_scores(question, outputs, weights, display)
        idxs = list(map(lambda x: x[0], scores))[:top_n]
        outputs = [outputs[idx] for idx in idxs]
        final_output = list(map(lambda x: self.post_processor.process(x), outputs))
        final_output = "\n".join(final_output)
        self.string += final_output
        if not gradio: return final_output.split("\n")
        else: return self.string

paraphraser = Paraphraser(model, paraqd, corrupter, post_processor)

display = True
n = 8
top_n = 2
num_return = 1
weights = [0.85, 0.15]


def paraphrase(sentence):
    return paraphraser.paraphrase(sentence, n=n, top_n=top_n, num_return=num_return, weights=weights)

