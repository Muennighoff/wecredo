## Whole Word Masking

### WWM for CN

### >>> Write your own tokenizer class extending HGFace
### Could take in as argument prepend text for translate etc tasks; returns decoder labels, decoder inputs, encoder inputs


# 1) Generate ref ids based on LTP tokenizer > prepare_ref
# 2) Generate mask for whole words
# 3) Implement the masking

from transformers import AutoTokenizer
import random
import numpy as np
from ltp import LTP

import tensorflow as tf


def random_word(tokens, ref_ids, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param ref_ids: list of int, 1 is where to place a mask
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, (token, ref_id) in enumerate(zip(tokens, ref_ids)):

        prob = random.random()

        if ref_id == 1:

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-100)

    return tokens, output_label

def _is_chinese_char(cp):
    """
    Checks whether CP is the codepoint of a CJK character.
    """
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def is_chinese(word):
    """
    Args:
      word: str
    """
    # word like '180' or '身高' or '神'
    for char in word:
        char = ord(char)
        if not _is_chinese_char(char):
            return 0
    return 1


def get_chinese_word(tokens):
    """
    Args:
      List[str]
    
    """
    word_set = set()

    for token in tokens:
        chinese_word = len(token) > 1 and is_chinese(token)
        if chinese_word:
            word_set.add(token)
    word_list = list(word_set)
    return word_list


def add_sub_symbol(bert_tokens, chinese_word_set):
    """
    Args:
      bert_tokens: List[str]
      chinese_word_set: set
    """
    if not chinese_word_set:
        return bert_tokens
    max_word_len = max([len(w) for w in chinese_word_set])

    bert_word = bert_tokens
    start, end = 0, len(bert_word)
    while start < end:
        single_word = True
        if is_chinese(bert_word[start]):
            l = min(end - start, max_word_len)
            for i in range(l, 1, -1):
                whole_word = "".join(bert_word[start : start + i])
                if whole_word in chinese_word_set:
                    for j in range(start + 1, start + i):
                        bert_word[j] = "##" + bert_word[j]
                    start = start + i
                    single_word = False
                    break
        if single_word:
            start += 1
    return bert_word

def prepare_ref(lines, ltp_tokenizer, bert_tokenizer):
    """
    
    Args:
      lines: List[str] - e.g. [text1, text2]
      ltp_tokenizer
      bert_tokenizer

    Returns:
      ref_ids: List[List[int], ...]
    
    """

    ltp_res = []

    for i in range(0, len(lines), 100):
        res = ltp_tokenizer.seg(lines[i : i + 100])[0]
        res = [get_chinese_word(r) for r in res]
        ltp_res.extend(res)
    assert len(ltp_res) == len(lines)

    bert_res = []
    for i in range(0, len(lines), 100):
        res = bert_tokenizer(lines[i : i + 100], add_special_tokens=True, truncation=True, max_length=512)
        bert_res.extend(res["input_ids"])
    assert len(bert_res) == len(lines)

    ref_ids = []
    for input_ids, chinese_word in zip(bert_res, ltp_res):

        input_tokens = []
        for id in input_ids:
            token = bert_tokenizer._convert_id_to_token(id)
            input_tokens.append(token)
        input_tokens = add_sub_symbol(input_tokens, chinese_word)
        ref_id = []
        # We only save pos of chinese subwords start with ##, which mean is part of a whole word.
        for i, token in enumerate(input_tokens):
            if token[:2] == "##":
                clean_token = token[2:]
                # save chinese tokens' pos
                if len(clean_token) == 1 and _is_chinese_char(ord(clean_token)):
                    ref_id.append(i)
        ref_ids.append(ref_id)

    assert len(ref_ids) == len(bert_res)

    return ref_ids


def cn_whole_word_mask(input_tokens, ref_ids, max_predictions=512, mlm_probability=0.15):
    """
    Masks whole words in CN based on the reference ids & the standard _whole_word_mask for BERT for one individual example.

    Args:
      input_tokens: List[str]
      ref_tokens: List[int]

    Returns:
      input_tokens: List[int]

    TODO:
      We could save the LTP dependency by copying the function from: https://github.com/HIT-SCIR/ltp/blob/c47b3f455c07c5dcc186f2b674efde8c67612baf/ltp/algorithms/maximum_forward_matching.py#L75
    """

    for i in range(len(input_tokens)):
        if i in ref_ids:
            # We move it back by -1 as the ref_ids start at 1, not 0
            input_tokens[i-1] = "##" + input_tokens[i-1]

    input_tokens = _whole_word_mask(input_tokens)

    return input_tokens



def _whole_word_mask(input_tokens, max_predictions=512, mlm_probability=0.15):
    """
    Get 0/1 labels for masked tokens with whole word mask proxy

    Args:
      input_tokens: List[str]

    Outputs:
      input_tokens: List[int]
    """

    cand_indexes = []
    for (i, token) in enumerate(input_tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue

        if len(cand_indexes) >= 1 and token.startswith("##"):
            cand_indexes[-1].append(i)

        else:
            cand_indexes.append([i])

    random.shuffle(cand_indexes)
    num_to_predict = min(max_predictions, max(1, int(round(len(input_tokens) * mlm_probability))))
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_lms.append(index)

    assert len(covered_indexes) == len(masked_lms)
    mask_labels = [1 if i in covered_indexes else 0 for i in range(len(input_tokens))]
    return mask_labels



class WWMTokenizer():
    def __init__(self, seq_len=512):
        """
        Constructs Huggingface CN tokenizer & other
        """

        self.tokenizer_cn = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.tokenizer_ltp = LTP("small")
        self.max_seq_length = seq_len

    def tokenize_pretraining(self, example):
        """
        Takes in an example & returns pretraining data

        Args:
            Example: dict with entry "text"

        Returns:
            Dict of TF Tensors

        """

        inputs = example['text']
    

        ref_ids = prepare_ref([inputs], self.tokenizer_ltp, self.tokenizer_cn)

        tokens = self.tokenizer_cn.tokenize(inputs)

        if len(tokens) > self.max_seq_length - 2:
            tokens = tokens[:(self.max_seq_length - 2)]
            ref_ids = ref_ids[:(self.max_seq_length - 2)]

        ref_ids = cn_whole_word_mask(tokens, ref_ids[0])
        tokens, labels = random_word(tokens, ref_ids, self.tokenizer_cn)

        tokens = ['[CLS]'] + tokens + ['[SEP]']
        lm_label_ids = ([-100] + labels + [-100])

        input_ids = self.tokenizer_cn.convert_tokens_to_ids(tokens)

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            attention_mask.append(0)
            token_type_ids.append(0)
            lm_label_ids.append(-100)

        assert len(input_ids) == self.max_seq_length
        assert len(attention_mask) == self.max_seq_length
        assert len(token_type_ids) == self.max_seq_length
        assert len(lm_label_ids) == self.max_seq_length


        outputs = {'input_ids': tf.constant(input_ids), 'attention_mask': tf.constant(attention_mask), 
                'token_type_ids': tf.constant(attention_mask), 'lm_label_ids': tf.constant(lm_label_ids)}

        return outputs

    def to_tf_dataset(self, dataset): 
        """
        Turns dataset into a TF compatible dataset
        """
        columns = ['input_ids', 'attention_mask', 'token_type_ids', 'lm_label_ids']
        dataset.set_format(type='tensorflow', columns=columns)

        return_types = {'input_ids':tf.int32, 'attention_mask':tf.int32, 
                      'token_type_ids':tf.int32, 'lm_label_ids':tf.int32}

        return_shapes = {'input_ids': tf.TensorShape([None]), 'attention_mask': tf.TensorShape([None]), 
                        'token_type_ids': tf.TensorShape([None]), 'lm_label_ids':tf.TensorShape([None])}

        ds = tf.data.Dataset.from_generator(lambda : dataset, return_types, return_shapes)
        return ds