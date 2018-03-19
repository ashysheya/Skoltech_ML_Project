#!/usr/bin/env python3
import collections
import numpy as np
import cityhash
import re
import os
import pickle
import pprint
import sys
import argparse
from tqdm import tqdm
import utils

###############################################################################
#                                                                             #
#                                INPUT DATA                                   #
#                                                                             #
###############################################################################

# Word: str
# Sentence: list of str
TaggedWord = collections.namedtuple('TaggedWord', ['text', 'tag'])
# TaggedSentence: list of TaggedWord
# Tags: list of TaggedWord
# TagLattice: list of Tags

TaggingQuality = collections.namedtuple('TaggingQuality', ['acc', 'f1'])


def tagging_quality(ref, out):
    """
    Compute tagging quality and return TaggingQuality object.
    """
    nwords = 0
    ncorrect = 0
    import itertools
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    i = 0
    for ref_sentence, out_sentence in itertools.zip_longest(ref, out):
        for ref_word, out_word in itertools.zip_longest(ref_sentence, out_sentence):
            nwords += 1
            ncorrect += ref_word.tag == out_word.tag
            if ref_word.tag != 'none' and out_word.tag != 'none':
                true_positive += 1
            elif ref_word.tag == 'none' and out_word.tag == 'none':
                true_negative += 1
            elif ref_word.tag == 'none' and out_word.tag != 'none':
                false_positive += 1
            else:
                false_negative += 1
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f_measure = 2.0*precision*recall/(precision + recall)
    return TaggingQuality(acc=ncorrect / nwords, f1=f_measure)


def read_tagged_sentences(path):
    """
    Read tagged sentences from file and return array of TaggedSentence.
    """
    sentences = [[]]
    with open(path, 'r') as fsent:
        for line in fsent:
            line = line.strip()
            if len(line) == 0 and len(sentences[-1]) > 0:
                sentences.append([])
            else:
                parts = line.split()
                word, tag = parts[0], parts[1]
                sentences[-1].append(TaggedWord(text=word, tag=tag))
    if len(sentences[-1]) == 0:
        del sentences[-1]
    return sentences


###############################################################################
#                                                                             #
#                             VALUE & UPDATE                                  #
#                                                                             #
###############################################################################


class Value:
    """
    Dense object that holds parameters.
    """

    def __init__(self, n):
        self.value = np.ones(n)
        self.n = n

    def dot(self, update):
        sum_update = 0
        for i, pos in enumerate(update.positions):
            sum_update += self.value[int(pos % self.n)] * update.values[i]
        return sum_update

    def assign(self, other):
        """
        self = other
        other is Value.
        """
        self.value = np.copy(other.value)
        self.n = other.n

    def assign_mul(self, coeff):
        """
        self = self * coeff
        coeff is float.
        """
        self.value *= coeff

    def assign_madd(self, x, coeff):
        """
        self = self + x * coeff
        x can be either Value or Update.
        coeff is float.
        """
        if isinstance(x, Value):
            self.value += x.value * coeff

        elif isinstance(x, Update):
            for i, pos in enumerate(x.positions):
                self.value[int(pos % self.n)] += x.values[i] * coeff


class Update:
    """
    Sparse object that holds an update of parameters.
    """

    def __init__(self, positions=None, values=None):
        """
        positions: array of int
        values: array of float
        """
        if positions is not None:
            self.positions = np.array(positions)
            self.values = np.array(values)

        else:
            self.positions = positions
            self.values = values

    def assign_mul(self, coeff):
        """
        self = self * coeff
        coeff: float
        """
        if self.positions is not None:
            self.values *= coeff

    def assign_madd(self, update, coeff):
        """
        self = self + update * coeff
        coeff: float
        """
        if self.positions is not None:
            self.positions = np.append(self.positions, update.positions)
            self.values = np.append(self.values, coeff * update.values)
        else:
            self.positions = np.copy(update.positions)
            self.values = coeff * update.values


###############################################################################
#                                                                             #
#                                  MODEL                                      #
#                                                                             #
###############################################################################


Features = Update


class LinearModel:
    """
    A thing that computes score and gradient for given features.
    """

    def __init__(self, n):
        self._params = Value(n)

    def params(self):
        return self._params

    def score(self, features):
        """
        features: Update
        """
        return self._params.dot(features)

    def gradient(self, features, score):
        return features


###############################################################################
#                                                                             #
#                                    HYPO                                     #
#                                                                             #
###############################################################################


Hypo = collections.namedtuple('Hypo', ['prev', 'pos', 'tagged_word', 'score'])
# prev: previous Hypo
# pos: position of word (0-based)
# tagged_word: tagging of source_sentence[pos]
# score: sum of scores over edges

###############################################################################
#                                                                             #
#                              FEATURE COMPUTER                               #
#                                                                             #
###############################################################################


def h(x):
    """
    Compute CityHash of any object.
    Can be used to construct features.
    """
    return cityhash.CityHash64(repr(x))


TaggerParams = collections.namedtuple('FeatureParams', [
    'src_window',
    'dst_order',
    'max_suffix',
    'beam_size',
    'nparams'
    ])


class FeatureComputer:
    def __init__(self, tagger_params, source_sentence):
        self._tagger_params = tagger_params
        self._source_sentence = source_sentence

    def compute_features(self, hypo):
        """
        Compute features for a given Hypo and return Update.
        """

        features = [(hypo.tagged_word.tag, 'current_word', hypo.tagged_word.text)]

        has_number = re.search('\d', hypo.tagged_word.text) is not None

        features.append((hypo.tagged_word.tag, 'has_number', has_number))

        has_upper = hypo.tagged_word.text != hypo.tagged_word.text.lower()

        features.append((hypo.tagged_word.tag, 'has_upper', has_upper))

        has_hyphen = re.search('-', hypo.tagged_word.text) is not None

        features.append((hypo.tagged_word.tag, 'has_hyphen', has_hyphen))

        for i in range(1, self._tagger_params.max_suffix + 1):
            features.append((hypo.tagged_word.tag, 'prefix_{}'.format(i),
                             hypo.tagged_word.text[:i]))

            features.append((hypo.tagged_word.tag, 'suffix_{}'.format(i),
                              hypo.tagged_word.text[-i:]))

        cur_hypo = hypo
        str_tags = []

        for i in range(self._tagger_params.dst_order):
            if cur_hypo.prev is not None:
                str_tags.append(cur_hypo.prev.tagged_word.tag)
                features.append((hypo.tagged_word.tag, 'prev_tags_{}'.format(len(str_tags)),
                                 '_'.join(str_tags[::-1])))
                cur_hypo = cur_hypo.prev
            else:
                str_tags.append('None')
                features.append((hypo.tagged_word.tag, 'prev_tags_{}'.format(len(str_tags)),
                                 '_'.join(str_tags[::-1])))

        for i in range(1, self._tagger_params.src_window + 1):
            if hypo.pos + i < len(self._source_sentence):
                features.append((hypo.tagged_word.tag, 'word_{}'.format(i),
                                   self._source_sentence[hypo.pos + i]))
            else:
                features.append((hypo.tagged_word.tag, 'word_{}'.format(i), 'None'))

            if hypo.pos - i >= 0:
                features.append((hypo.tagged_word.tag, 'word_{}'.format(-i),
                                   self._source_sentence[hypo.pos - i]))
            else:
                features.append((hypo.tagged_word.tag, 'word_{}'.format(-i), 'None'))

        pos = np.unique(np.array([h(f) % self._tagger_params.nparams for f in features]))

        return Update(positions=pos, values=np.ones(pos.shape[0]))



###############################################################################
#                                                                             #
#                                BEAM SEARCH                                  #
#                                                                             #
###############################################################################


class BeamSearchTask:
    """
    An abstract beam search task. Can be used with beam_search() generic 
    function.
    """

    def __init__(self, tagger_params, source_sentence, model, tags):
        self._tagger_params = tagger_params
        self._source_sentence = source_sentence
        self._model = model
        self._tags = tags
        self._feature_computer = FeatureComputer(tagger_params=tagger_params,
                                                 source_sentence=source_sentence)

    def total_num_steps(self):
        """
        Number of hypotheses between beginning and end (number of words in
        the sentence).
        """
        return len(self._source_sentence)

    def beam_size(self):
        return self._tagger_params.beam_size

    def expand(self, hypo):
        """
        Given Hypo, return a list of its possible expansions.
        'hypo' might be None -- return a list of initial hypos then.

        Compute hypotheses' scores inside this function!
        """
        possible_hypos = []
        if hypo is not None:
            for tag in self._tags:
                new_hypo = Hypo(prev=hypo, pos=hypo.pos + 1,
                                tagged_word=TaggedWord(text=self._source_sentence[hypo.pos + 1],
                                                       tag=tag),
                                score=-1)

                new_feature = self._feature_computer.compute_features(new_hypo)
                score = self._model.score(new_feature)
                possible_hypos.append(new_hypo._replace(score=score + hypo.score))
        else:
            for tag in self._tags:
                new_hypo = Hypo(prev=None, pos=0,
                                tagged_word=TaggedWord(text=self._source_sentence[0],
                                                       tag=tag),
                                score=-1)
                new_feature = self._feature_computer.compute_features(new_hypo)
                score = self._model.score(new_feature)
                possible_hypos.append(new_hypo._replace(score=score))
        return possible_hypos


def beam_search(beam_search_task):
    """
    Return list of stacks.
    Each stack contains several hypos, sorted by score in descending 
    order (i.e. better hypos first).
    """
    num_steps = beam_search_task.total_num_steps()
    beam_size = beam_search_task.beam_size()
    stacks = []
    for i in range(num_steps):
        if len(stacks) == 0:
            ans = beam_search_task.expand(None)
            ans = sorted(ans, reverse=True, key=lambda el: el.score)[:beam_size]
            stacks.append(ans)

        else:
            cur_stack = []
            for cur_hypo in stacks[-1]:
                cur_stack += beam_search_task.expand(cur_hypo)
            stacks.append(sorted(cur_stack, reverse=True, key=lambda el: el.score)[:beam_size])
    return stacks


###############################################################################
#                                                                             #
#                            OPTIMIZATION TASKS                               #
#                                                                             #
###############################################################################


class OptimizationTask:
    """
    Optimization task that can be used with sgd().
    """

    def params(self):
        """
        Parameters which are optimized in this optimization task.
        Return Value.
        """
        raise NotImplementedError()

    def loss_and_gradient(self, golden_sentence):
        """
        Return (loss, gradient) on a specific example.

        loss: float
        gradient: Update
        """
        raise NotImplementedError()


class StructuredPerceptronOptimizationTask(OptimizationTask):
    def __init__(self, tagger_params, tags):
        self.tagger_params = tagger_params
        self.model = LinearModel(tagger_params.nparams)
        self.tags = tags

    def params(self):
        return self.model.params()

    def loss_and_gradient(self, golden_sentence):
        golden_text = [golden_tagged_word.text for golden_tagged_word in golden_sentence]
        # Do beam search.
        beam_search_task = BeamSearchTask(
            self.tagger_params, 
            golden_text,
            self.model, 
            self.tags
            )

        stacks = beam_search(beam_search_task)
        # Compute chain of golden hypos (and their scores!).
        golden_hypo = None
        feature_computer = FeatureComputer(tagger_params=self.tagger_params,
                                           source_sentence=golden_text)

        score_diff = []
        golden_hypos = []

        for i in range(len(golden_sentence)):
            new_golden_hypo = Hypo(prev=golden_hypo, pos=i, tagged_word=golden_sentence[i],
                                   score=-1)
            new_feature = feature_computer.compute_features(new_golden_hypo)
            score = self.model.score(new_feature)
            if golden_hypo is None:
                new_golden_hypo = new_golden_hypo._replace(score=score)
            else:
                new_golden_hypo = new_golden_hypo._replace(score=score + golden_hypo.score)
            golden_hypo = new_golden_hypo
            golden_hypos.append(golden_hypo)
            score_diff.append(golden_hypo.score - stacks[i][-1].score)

        score_diff = np.array(score_diff)

        idx = np.argmin(score_diff)
        delta = np.min(score_diff)
        # Find where to update.

        if delta < 0 and idx < len(golden_sentence) - 1:
            golden_head = golden_hypos[idx]
            rival_head = stacks[idx][0]

        else:
            golden_head = golden_hypos[-1]
            rival_head = stacks[-1][0]


        # Compute gradient.
        grad = Update()
        while golden_head and rival_head:
            rival_features = feature_computer.compute_features(rival_head)
            grad.assign_madd(self.model.gradient(rival_features, score=None), 1)

            golden_features = feature_computer.compute_features(golden_head)
            grad.assign_madd(self.model.gradient(golden_features, score=None), -1)

            golden_head = golden_head.prev
            rival_head = rival_head.prev

        return grad
        

###############################################################################
#                                                                             #
#                                    SGD                                      #
#                                                                             #
###############################################################################


SGDParams = collections.namedtuple('SGDParams', [
    'epochs',
    'learning_rate',
    'minibatch_size',
    'average'  # bool or int
    ])


def make_batches(dataset, minibatch_size):
    """
    Make list of batches from a list of examples.
    """
    size_data = len(dataset)
    permutation = np.random.permutation(size_data)
    num_batches = int(np.ceil(size_data / minibatch_size))
    batches = []
    for i in range(num_batches):
        batches.append([])
        for j in range(i * minibatch_size, min((i + 1) * minibatch_size, permutation.shape[0])):
            batches[-1].append(dataset[permutation[j]])
    return batches


def sgd(sgd_params, optimization_task, dataset, after_each_epoch_fn):
    """
    Run (averaged) SGD on a generic optimization task. Modify optimization
    task's parameters.

    After each epoch (and also before and after the whole training),
    run after_each_epoch_fn().
    """
    after_each_epoch_fn()
    averaged = Value(optimization_task.params().n)
    averaged.assign_mul(0.0)
    num_grad = 0
    for epoch in tqdm(range(sgd_params.epochs)):
        batches = make_batches(dataset, sgd_params.minibatch_size)
        for i, batch in tqdm(enumerate(batches)):
            grad = Update()
            for sentence in batch:
                grad.assign_madd(update=optimization_task.loss_and_gradient(sentence), coeff=1.0)
            grad.assign_mul(1.0 / len(batch))
            optimization_task.params().assign_madd(x=grad, coeff=-sgd_params.learning_rate)

            if i % sgd_params.average == 0:
                averaged.assign_madd(x=optimization_task.params(), coeff=1.0)
                num_grad += 1
        after_each_epoch_fn()
    averaged.assign_mul(1.0 / num_grad)
    optimization_task.params().assign(averaged)
    pickle.dump(optimization_task.params(), open('averaged.npz', 'wb'))
    #after_each_epoch_fn()


###############################################################################
#                                                                             #
#                                    MAIN                                     #
#                                                                             #
###############################################################################


# - Train - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def TRAIN_add_cmdargs(subp):
    p = subp.add_parser('train')

    p.add_argument('--dataset',
        help='train dataset', default='train')
    p.add_argument('--dataset-dev',
        help='dev dataset', default='val')
    p.add_argument('--tags',
        help='tags file', default='tags')
    p.add_argument('--model',
        help='NPZ model', type=str, default='model.npz')
    p.add_argument('--sgd-epochs',
        help='SGD number of epochs', type=int, default=15)
    p.add_argument('--sgd-learning-rate',
        help='SGD learning rate', type=float, default=1.0)
    p.add_argument('--sgd-minibatch-size',
        help='SGD minibatch size (in sentences)', type=int, default=1)
    p.add_argument('--sgd-average',
        help='SGD average every N batches', type=int, default=10)
    p.add_argument('--tagger-src-window',
        help='Number of context words in input sentence to use for features',
        type=int, default=2)
    p.add_argument('--tagger-dst-order',
        help='Number of context tags in output tagging to use for features',
        type=int, default=3)
    p.add_argument('--tagger-max-suffix',
        help='Maximal number of prefix/suffix letters to use for features',
        type=int, default=4)
    p.add_argument('--beam-size',
        help='Beam size', type=int, default=4)
    p.add_argument('--nparams',
        help='Parameter vector size', type=int, default=2**22)

    return 'train'

def TRAIN(cmdargs):
    # Beam size.
    optimization_task_cls = StructuredPerceptronOptimizationTask

    print('Reading train data...')
    # Parse cmdargs.
    if not os.path.exists(cmdargs.dataset):
        utils.dataset_reader()

    dataset = read_tagged_sentences(cmdargs.dataset)
    tags = utils.read_tags(cmdargs.tags)

    print('Reading validation data...')
    dataset_dev = read_tagged_sentences(cmdargs.dataset_dev)

    params = None
    if os.path.exists(cmdargs.model):
        params = pickle.load(open(cmdargs.model, 'rb'))
    sgd_params = SGDParams(
        epochs=cmdargs.sgd_epochs,
        learning_rate=cmdargs.sgd_learning_rate,
        minibatch_size=cmdargs.sgd_minibatch_size,
        average=cmdargs.sgd_average
        )
    tagger_params = TaggerParams(
        src_window=cmdargs.tagger_src_window,
        dst_order=cmdargs.tagger_dst_order,
        max_suffix=cmdargs.tagger_max_suffix,
        beam_size=cmdargs.beam_size,
        nparams=cmdargs.nparams
        )

    # Load optimization task
    optimization_task = optimization_task_cls(tagger_params=tagger_params, tags=tags)
    if params is not None:
        print('\n\nLoading parameters from %s\n\n' % cmdargs.model)
        optimization_task.params().assign(params)

    # Validation.
    def after_each_epoch_fn():
        model = LinearModel(cmdargs.nparams)
        model.params().assign(optimization_task.params())
        tagged_sentences = tag_sentences(dataset=dataset_dev, model=model,
                                         tagger_params=tagger_params, tags=tags)

        q = pprint.pformat(tagging_quality(out=tagged_sentences, ref=dataset_dev))
        print()
        print(q)
        print()

        # Save parameters.
        print('\n\nSaving parameters to %s\n\n' % cmdargs.model)
        pickle.dump(optimization_task.params(), open(cmdargs.model, 'wb'))

    # Run SGD.
    sgd(sgd_params, optimization_task, dataset, after_each_epoch_fn)


# - Test  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def TEST_add_cmdargs(subp):
    p = subp.add_parser('test')

    p.add_argument('--tags',
        help='tags file', type=str, default='tags')
    p.add_argument('--dataset',
        help='test dataset', default='test')
    p.add_argument('--model',
        help='NPZ model', type=str, default='averaged.npz')
    p.add_argument('--tagger-src-window',
        help='Number of context words in input sentence to use for features',
        type=int, default=2)
    p.add_argument('--tagger-dst-order',
        help='Number of context tags in output tagging to use for features',
        type=int, default=3)
    p.add_argument('--tagger-max-suffix',
        help='Maximal number of prefix/suffix letters to use for features',
        type=int, default=4)
    p.add_argument('--beam-size',
        help='Beam size', type=int, default=4)

    return 'test'


def tag_sentences(dataset, model, tagger_params, tags):
    """
    Tag all sentences in dataset. Dataset is a list of TaggedSentence; while 
    tagging, ignore existing tags.
    """
    tagged_dataset = []
    for sentence in tqdm(dataset):
        tagged_dataset.append([])
        text = [sent.text for sent in sentence]
        best_hypo = beam_search(BeamSearchTask(model=model, tagger_params=tagger_params,
                                               source_sentence=text, tags=tags))[-1][0]

        while best_hypo:
            tagged_dataset[-1].append(best_hypo.tagged_word)
            best_hypo = best_hypo.prev
        tagged_dataset[-1] = tagged_dataset[-1][::-1]
    return tagged_dataset


def TEST(cmdargs):
    # Parse cmdargs.
    tags = utils.read_tags(cmdargs.tags)
    dataset = read_tagged_sentences(cmdargs.dataset)
    params = pickle.load(open(cmdargs.model, 'rb'))
    tagger_params = TaggerParams(
        src_window=cmdargs.tagger_src_window,
        dst_order=cmdargs.tagger_dst_order,
        max_suffix=cmdargs.tagger_max_suffix,
        beam_size=cmdargs.beam_size,
        nparams=params.value.shape[0]
        )

    # Load model.
    model = LinearModel(params.value.shape[0])
    model.params().assign(params)

    # Tag all sentences.
    tagged_sentences = tag_sentences(dataset=dataset, tagger_params=tagger_params,
                                     tags=tags, model=model)

    # Measure and print quality.
    q = pprint.pformat(tagging_quality(out=tagged_sentences, ref=dataset))
    print(q, file=sys.stderr)


# - Main  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def main():

    # Create parser.
    p = argparse.ArgumentParser('tagger.py')
    subp = p.add_subparsers(dest='cmd')

    # Add subcommands.
    train = TRAIN_add_cmdargs(subp)
    test = TEST_add_cmdargs(subp)

    # Parse.
    cmdargs = p.parse_args()

    # Run.
    if cmdargs.cmd == train:
        TRAIN(cmdargs)
    elif cmdargs.cmd == test:
        TEST(cmdargs)
    else:
        p.error('No command')

if __name__ == '__main__':
    main()
