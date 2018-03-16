import numpy as np
import glob
import subprocess
import json
import re

_TOKEN_SPECIAL_CODES = {'__unknown__': 0, '__padding__': 1, '__begin__': 2,
                       '__end__': 3}
_SYMBOL_SPECIAL_CODES = {'__padding__': 0, '__begin__': 1, '__end__': 2}

_MYSTEM_TO_WORD2VEC_POS_TAGS_MAPPING = {
    'A': 'ADJ',
    'ADV': 'ADV',
    'ADVPRO': 'ADV',
    'ANUM': 'ADJ',
    'APRO': 'PRON',
    'COM': 'X',
    'CONJ': 'CCONJ',
    'INTJ': 'INTJ',
    'NUM': 'NUM',
    'PART': 'PART',
    'PR': 'ADP',
    'S': 'NOUN',
    'SPRO': 'PRON',
    'V': 'VERB'
}

def load_sentences_and_labels(directory_name, return_all_tags=False):

    # https://github.com/dialogue-evaluation/factRuEval-2016
    # devset for training/validation
    # testset for final testing

    token_filenames = glob.glob('{}/*.tokens'.format(directory_name))
    span_filenames = glob.glob('{}/*.spans'.format(directory_name))

    token_filenames = list(sorted(token_filenames))
    span_filenames = list(sorted(span_filenames))

    sentences = []
    labels = []

    if return_all_tags:
        tags = set()

    for token_filename, span_filename in zip(token_filenames, span_filenames):

        token_labels = {}

        with open(span_filename, 'r') as span_file:
            for line in span_file:

                splitted_line = line.split()

                label = splitted_line[1]
                first_token_index = int(splitted_line[4])
                tokens_count = int(splitted_line[5])

                for token_index in range(first_token_index, first_token_index + tokens_count):
                    token_labels[token_index] = label # add IOB format???
        
        current_sentence = []
        current_sentence_labels = []

        with open(token_filename, 'r') as token_file:
            for line in token_file:

                splitted_line = line.split()
                
                if len(splitted_line) == 0:

                    sentences.append(current_sentence)
                    labels.append(current_sentence_labels)

                    current_sentence = []
                    current_sentence_labels = []

                    continue

                token_index = int(splitted_line[0])
                token = splitted_line[-1]

                if token_index in token_labels:
                    label = token_labels[token_index]
                else:
                    label = 'none'

                if return_all_tags:
                    tags.add(label)
                current_sentence.append(token)
                current_sentence_labels.append(label)

    if return_all_tags:
        return np.array(sentences), np.array(labels), np.array(list(tags))
    return np.array(sentences), np.array(labels)

def _get_normalized_sentence(sentence, add_w2v_tag=True):

    # Requires mystem executable
    
    with open('temp_input.txt', 'w') as input_file:
        input_file.write(' '.join(sentence) + '\n')

    command = './mystem -ndic --format json temp_input.txt temp_output.txt'
    subprocess.call(command.split())

    parsed_sentence = []

    with open('temp_output.txt', 'r') as output_file:
        for line in output_file:

            parsed_token = json.loads(line)

            parsed_token['text'] = parsed_token['text'].strip()

            splitted_text = parsed_token['text'].split()

            if len(splitted_text) == 0:
                continue

            if len(splitted_text) == 1:
                parsed_sentence.append(parsed_token)
                continue

            for chunk in splitted_text:
                parsed_sentence.append({'text': chunk})

    offset = 0

    updated_parsed_sentence = []

    for index, token in enumerate(sentence):
        
        parsed_token = parsed_sentence[index + offset]

        parsed_token_text = parsed_token['text']
        
        if token == parsed_token_text:
            updated_parsed_sentence.append(parsed_token)
            continue

        while token != parsed_token_text:
            offset += 1
            parsed_token_text += parsed_sentence[index + offset]['text']

        updated_parsed_sentence.append({'text': parsed_token_text})

    parsed_sentence = updated_parsed_sentence
        
    normalized_sentence = []

    for parsed_token in parsed_sentence:

        if 'analysis' in parsed_token and len(parsed_token['analysis']) > 0:

            mystem_pos_tag = re.findall('^[^=,]*', parsed_token['analysis'][0]['gr'])[0]
            word2vec_pos_tag = _MYSTEM_TO_WORD2VEC_POS_TAGS_MAPPING[mystem_pos_tag]
            normalized_token = parsed_token['analysis'][0]['lex']
            if add_w2v_tag:
                normalized_token += '_' + word2vec_pos_tag

        else:
            normalized_token = parsed_token['text']
            if add_w2v_tag:
                normalized_token += '_X'

        normalized_sentence.append(normalized_token)

    return normalized_sentence

def get_normalized_sentences(sentences, add_w2v_tag=True):

    print('Sentence normalization')

    normalized_sentences = []

    for index, sentence in enumerate(sentences):

        normalized_sentences.append(_get_normalized_sentence(sentence, add_w2v_tag))

        if (index + 1)%100 == 0:
            print('{} sentences normalized'.format(index + 1))

    return normalized_sentences

def load_word2vec():

    # http://rusvectores.org/static/models/rusvectores4/ruwikiruscorpora/ruwikiruscorpora_upos_skipgram_300_2_2018.vec.gz
    
    word2vec_filename = 'ruwikiruscorpora_superbigrams_2_1_2.vec'

    word2vec = {}

    with open(word2vec_filename, 'r') as word2vec_file:
        for line in word2vec_file:
            splitted_line = line.split()
            word2vec[splitted_line[0]] = np.array([float(x) for x in splitted_line[1:]])

    return word2vec

def build_token_embeddings_tensor(token_codes, word2vec, word2vec_dimension):

    embedding_array = np.zeros((len(token_codes), word2vec_dimension))

    for token, code in token_codes.items():

        if token in word2vec:
            embedding_array[code] = word2vec[token]
        else:
            embedding_array[code] = np.random.normal(size=word2vec_dimension)

    return torch.Tensor(embedding_array)

def _get_codes(sentences, normalized_sentences, labels):

    symbol_codes = dict(_SYMBOL_SPECIAL_CODES)
    token_codes = dict(_TOKEN_SPECIAL_CODES)
    label_codes = {}

    for sentence, normalized_sentence, sentence_labels in zip(sentences,
                                                              normalized_sentences,
                                                              labels):
        for token, normalized_token, label in zip(sentence, normalized_sentence,
                                                  sentence_labels):

            if normalized_token not in token_codes:
                token_codes[normalized_token] = len(token_codes)

            if label not in label_codes:
                label_codes[label] = len(label_codes)

            for symbol in token:
                if symbol not in symbol_codes:
                    symbol_codes[symbol] = len(symbol_codes)

    return symbol_codes, token_codes, label_codes

def _get_max_sentence_and_token_lengths(sentences):

    max_sentence_length = 0
    max_token_length = 0

    for sentence in sentences:

        max_sentence_length = max(max_sentence_length, len(sentence))
        
        for token in sentence:
            max_token_length = max(max_token_length, len(token))

    return max_sentence_length, max_token_length

def encode_sentences_and_labels(sentences, normalized_sentences, labels, codes=None,
                                dimensions=None):

    if codes is None:
        codes = _get_codes(sentences, normalized_sentences, labels)
        return_codes = True
    else:
        return_codes = False

    symbol_codes, token_codes, label_codes = codes

    if dimensions is None:

        max_sentence_length, max_token_length = \
                _get_max_sentence_and_token_lengths(sentences)

        dimensions = (max_sentence_length + 2, max_token_length + 2)

        return_dimensions = True

    else:
        return_dimensions = False

    sentence_dimension, token_dimension = dimensions

    encoded_symbols = np.full((len(sentences), sentence_dimension*token_dimension),
                              symbol_codes['__padding__'], dtype=int)
    encoded_tokens = np.full((len(sentences), sentence_dimension),
                             token_codes['__padding__'], dtype=int)
    encoded_labels = np.full((len(sentences), sentence_dimension),
                             label_codes['none'], dtype=int)

    for sentence_index, sentence in enumerate(sentences):

        encoded_tokens[sentence_index, 0] = token_codes['__begin__']

        if len(sentence) + 1 < sentence_dimension:
            encoded_tokens[sentence_index, len(sentence) + 1] = token_codes['__end__']

        for token_index, (token, normalized_token) in \
                enumerate(zip(sentences[sentence_index],
                              normalized_sentences[sentence_index])):

            if token_index + 1 >= sentence_dimension:
                break
    
            label = labels[sentence_index][token_index]

            if normalized_token in token_codes:
                token_code = token_codes[normalized_token]
            else:
                token_code = token_codes['__unknown__']
            
            encoded_tokens[sentence_index, token_index + 1] = token_code
            encoded_labels[sentence_index, token_index + 1] = label_codes[label]

            symbol_index_offset = (token_index + 1)*token_dimension

            encoded_symbols[sentence_index, symbol_index_offset] = \
                    symbol_codes['__begin__']
            encoded_symbols[sentence_index, symbol_index_offset + len(token) + 1] = \
                    symbol_codes['__end__']

            for symbol_index, symbol in enumerate(token):

                if symbol_index + 1 >= token_dimension:
                    break

                if symbol in symbol_codes:
                    symbol_code = symbol_codes[symbol]
                else:
                    # c'est le kostyl'
                    symbol_code = symbol_codes['.']

                encoded_symbols[sentence_index, symbol_index_offset + symbol_index + 1] = \
                        symbol_code

    result = ()

    if return_codes:
        result += codes

    if return_dimensions:
        result += dimensions
        
    result += (encoded_symbols, encoded_tokens, encoded_labels)

    return result