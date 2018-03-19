import utils
import numpy as np


class Vocabulary:
    def __init__(self, tokens, bos='__begin__', eos='__end__', unk='__unknown__', use_w2v=True,
                 w2v_dim=300):
        self.token_to_idx = {token: i for i, token in enumerate(tokens)}
        self.use_w2v = use_w2v
        self.bos = bos
        self.eos = eos
        if unk is not None:
            self.unk = unk
            self.unk_idx = self.token_to_idx[unk]
        self.bos_idx = self.token_to_idx[bos]
        self.eos_idx = self.token_to_idx[eos]

        self.idx_to_token = {i: token for i, token in enumerate(tokens)}
        if use_w2v:
            word2vec = utils.load_word2vec('ruwikiruscorpora_upos_skipgram_300_2_2018.vec')
            self.word2vec_tokens = np.zeros((len(tokens), w2v_dim))
            self.w2v_dim = w2v_dim
            for token, idx in self.token_to_idx.items():
                if token in word2vec:
                    self.word2vec_tokens[idx] = word2vec[token]
                else:
                    self.word2vec_tokens[idx] = np.random.normal(w2v_dim)

    def __len__(self):
        return len(self.token_to_idx)

    @classmethod
    def from_lines(cls, sentences, use_w2v=True, unk=None):
        tokens = set()
        for sentence in sentences:
            for word in sentence:
                tokens.add(word)
        if unk is not None:
            tokens = ['__begin__', '__end__', '__unknown__'] + list(tokens)
        else:
            tokens = ['__begin__', '__end__'] + list(tokens)
        return Vocabulary(tokens, unk=unk, use_w2v=use_w2v)

    def to_matrix(self, sentences):
        max_len = max(map(len, sentences)) + 2
        matrix = np.zeros((len(sentences), max_len)) + self.eos_idx

        if self.use_w2v:
            w2v_matrix = np.zeros((len(sentences), max_len, self.w2v_dim)) + \
                         self.word2vec_tokens[self.eos_idx]
            w2v_matrix[:, 0] = self.word2vec_tokens[self.bos_idx]

        matrix[:, 0] = self.bos_idx
        for i, sentence in enumerate(sentences):
            idx_sentence = [self.token_to_idx[word] if word in self.token_to_idx else self.unk_idx
                            for word in sentence]
            matrix[i, 1: len(idx_sentence) + 1] = idx_sentence

            if self.use_w2v:
                w2v_sentence = [self.word2vec_tokens[idx] for idx in idx_sentence]
                w2v_matrix[i, 1: len(idx_sentence) + 1] = w2v_sentence
        if self.use_w2v:
            return matrix, w2v_matrix
        return matrix, None

    def to_lines(self, matrix):
        lines = []
        for line_ix in map(list, matrix):
            if line_ix[0] == self.bos_idx:
                line_ix = line_ix[1:]
            if self.eos_idx in line_ix:
                line_ix = line_ix[:line_ix.index(self.eos_idx)]
            line = [self.idx_to_token[idx] for idx in line_ix]
            lines.append(line)
        return lines
