import tensorflow as tf
import keras.layers as L


class AttentionTranslationModel:
    """
    Class for training Model with attention.
    """
    def __init__(self, name, inp_voc, out_voc,
                 emb_size, hid_size, mode='concat',
                 rnn_type='LSTM', is_bidir=False):
        '''
        Initialize model.
        :param name: name for variable scope
        :param inp_voc: vocabulary of input tokens
        :param out_voc: vocabulary of output tokens
        :param emb_size: word embedding size
        :param hid_size: hidden size in reccurent cells
        :param mode: attention type: concat, linear, general
        :param rnn_type: type of RNN cell - GRU, LSTM, RNN
        :param is_bidir: whether to use bidirectional reccurent cells
        '''
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hid_size = hid_size
        self.is_bidir = is_bidir
        self.rnn_type = rnn_type

        with tf.variable_scope(name):
            self.emb_inp = L.Dense(emb_size, input_shape=(None, inp_voc.w2v_dim))
            self.emb_out = L.Embedding(len(out_voc), emb_size)
            if self.is_bidir:
                if rnn_type == 'LSTM':
                    self.enc0_fw = tf.nn.rnn_cell.BasicLSTMCell(hid_size // 2)
                    self.enc0_bw = tf.nn.rnn_cell.BasicLSTMCell(hid_size // 2)
                    self.dec0 = tf.nn.rnn_cell.BasicLSTMCell(hid_size)
                elif rnn_type == 'GRU':
                    self.enc0_fw = tf.nn.rnn_cell.GRUCell(hid_size // 2)
                    self.enc0_bw = tf.nn.rnn_cell.GRUCell(hid_size // 2)
                    self.dec0 = tf.nn.rnn_cell.GRUCell(hid_size)
                else:
                    self.enc0_fw = tf.nn.rnn_cell.BasicRNNCell(hid_size // 2)
                    self.enc0_bw = tf.nn.rnn_cell.BasicRNNCell(hid_size // 2)
                    self.dec0 = tf.nn.rnn_cell.BasicRNNCell(hid_size)
            else:
                if rnn_type == 'LSTM':
                    self.enc0 = tf.nn.rnn_cell.BasicLSTMCell(hid_size)
                    self.dec0 = tf.nn.rnn_cell.BasicLSTMCell(hid_size)
                elif rnn_type == 'GRU':
                    self.enc0 = tf.nn.rnn_cell.GRUCell(hid_size)
                    self.dec0 = tf.nn.rnn_cell.GRUCell(hid_size)
                else:
                    self.dec0 = tf.nn.rnn_cell.BasicRNNCell(hid_size)
                    self.enc0 = tf.nn.rnn_cell.BasicRNNCell(hid_size)

            if rnn_type == 'LSTM':
                self.dec_start = (L.Dense(hid_size, name='dec_start_c'),
                                  L.Dense(hid_size, name='dec_start_h'))
            else:
                self.dec_start = L.Dense(hid_size, name='dec_start')

            self.logits = L.Dense(len(out_voc), name='logits')
            self.attention = AttentionLayer(name='attention_model', enc_hid_size=hid_size,
                                            dec_hid_size=hid_size, hid_size=hid_size,
                                            mode=mode)
            self.to_hidden = L.Dense(hid_size, activation=tf.tanh, input_dim=hid_size * 2,
                                     name='to_hidden')

            # run on dummy output to .build all layers (and therefore create weights)
            inp_matrix = tf.placeholder('int32', [None, None], 'tmp_input')
            out = tf.placeholder('int32', [None, None])
            inp = tf.placeholder('float32', [None, None, inp_voc.w2v_dim])
            with tf.variable_scope('enc'):
                h0, _ = self.encode(inp, inp_matrix)
            with tf.variable_scope('dec'):
                h1_state, h1_out = self.decode(h0, out[:, 0])
                h2 = self.get_next_hidden_state(h1_out, h1_out)
                logits = self.logits(h2)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)

    def encode(self, inp, inp_matrix):
        """
        Takes symbolic input sequence, computes initial state
        :param inp: matrix of input tokens [batch, time]
        :return: a list of initial decoder state tensors
        """
        inp_lengths = infer_length(inp_matrix, self.inp_voc.eos_idx)
        inp_emb = self.emb_inp(inp)

        if self.is_bidir:
            all_hidden_states, enc_last = tf.nn.bidirectional_dynamic_rnn(
                self.enc0_fw, self.enc0_bw, inp_emb,
                sequence_length=inp_lengths,
                dtype=inp_emb.dtype)

            all_hidden_states = tf.concat([all_hidden_states[0], all_hidden_states[1]], axis=2)
            if self.rnn_type == 'LSTM':
                c_concat = tf.concat([enc_last[0][0], enc_last[1][0]], axis=1)
                h_concat = tf.concat([enc_last[0][1], enc_last[1][1]], axis=1)

                dec_start = tf.nn.rnn_cell.LSTMStateTuple(self.dec_start[0](c_concat),
                                                          self.dec_start[1](h_concat))
            else:
                h_concat = tf.concat([enc_last[0], enc_last[1]], axis=1)
                dec_start = self.dec_start(h_concat)

        else:
            all_hidden_states, enc_last = tf.nn.dynamic_rnn(self.enc0, inp_emb,
                                                            sequence_length=inp_lengths,
                                                            dtype=inp_emb.dtype)
            if self.rnn_type == 'LSTM':
                dec_start = tf.nn.rnn_cell.LSTMStateTuple(self.dec_start[0](enc_last[0]),
                                                          self.dec_start[1](enc_last[1]))
            else:
                dec_start = self.dec_start(enc_last)

        return [dec_start], all_hidden_states

    def decode(self, prev_state, prev_tokens):
        """
        Takes previous decoder state and tokens, returns new state and logits
        :param prev_state: a list of previous decoder state tensors
        :param prev_tokens: previous output tokens, an int vector of [batch_size]
        :return: a list of next decoder state tensors, a tensor of logits [batch,n_tokens]
        """

        [prev_dec] = prev_state
        prev_emb = self.emb_out(prev_tokens[:, None])[:, 0]
        new_dec_out, new_dec_state = self.dec0(prev_emb, prev_dec)

        return [new_dec_state], new_dec_out

    def get_next_hidden_state(self, context, prev_state):
        return self.to_hidden(tf.concat([context, prev_state], axis=1))

    def symbolic_score(self, inp, inp_matrix, out, eps=1e-30):
        first_state, all_enc_hidden_states = self.encode(inp, inp_matrix)
        inp_mask = infer_mask(inp_matrix, self.inp_voc.eos_idx)
        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size], self.out_voc.bos_idx)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + eps)

        def step(blob, y_prev):
            h_prev = blob[:-1]
            h_current, dec_out = self.decode(h_prev, y_prev)
            context, _ = self.attention(all_enc_hidden_sates=all_enc_hidden_states,
                                        prev_state=dec_out,
                                        inp_mask=inp_mask)
            h_with_context = self.get_next_hidden_state(context, dec_out)
            logits = self.logits(h_with_context)
            return list(h_current) + [logits]

        results = tf.scan(step, initializer=list(first_state)+[first_logits],
                          elems=tf.transpose(out))


        logits_seq = results[-1]

        logits_seq = tf.concat((first_logits[None], logits_seq), axis=0)

        logits_seq = tf.transpose(logits_seq, [1, 0, 2])

        return tf.nn.log_softmax(logits_seq)

    def symbolic_translate(self, inp, inp_matrix, greedy=False, max_len=None, eps=1e-30):
        first_state, all_hidden_states = self.encode(inp, inp_matrix)

        inp_mask = infer_mask(inp_matrix, self.inp_voc.eos_idx)

        batch_size = tf.shape(inp)[0]
        bos = tf.fill([batch_size], self.out_voc.bos_idx)
        first_logits = tf.log(tf.one_hot(bos, len(self.out_voc)) + eps)
        max_len = tf.reduce_max(tf.shape(inp)[1])*2

        def step(blob, t):
            h_prev, y_prev = blob[:-2], blob[-1]
            h_current, dec_out = self.decode(h_prev, y_prev)
            context, _ = self.attention(all_enc_hidden_sates=all_hidden_states, prev_state=dec_out,
                                        inp_mask=inp_mask)  # [batch_size, enc_hid_size]

            h_with_context = self.get_next_hidden_state(context, dec_out)
            logits = self.logits(h_with_context)
            y_new = tf.argmax(logits, axis=-1) if greedy else tf.multinomial(logits, 1)[:, 0]
            return list(h_current) + [logits, tf.cast(y_new, y_prev.dtype)]

        results = tf.scan(step, initializer=list(first_state) + [first_logits, bos],
                          elems=[tf.range(max_len)])

        logits_seq, out_seq = results[-2], results[-1]

        logits_seq = tf.concat((first_logits[None],logits_seq),axis=0)
        out_seq = tf.concat((bos[None], out_seq), axis=0)

        logits_seq = tf.transpose(logits_seq, [1, 0, 2])
        out_seq = tf.transpose(out_seq)

        return out_seq, tf.nn.log_softmax(logits_seq)


class AttentionLayer:
    def __init__(self, name, enc_hid_size, dec_hid_size, hid_size, activ=tf.tanh, mode='concat'):
        """
        A basic layer that computes attention weights and response
        """
        self.name = name
        self.enc_size = enc_hid_size
        self.dec_size = dec_hid_size
        self.hid_size = hid_size
        self.activ = activ
        self.mode = mode

        with tf.variable_scope(name):
            if self.mode == 'concat':
                self.w_att = tf.get_variable('w_att', shape=[dec_hid_size, hid_size])
                self.u_att = tf.get_variable('u_att', shape=[enc_hid_size, hid_size])
                self.vec = tf.get_variable('vec', shape=[hid_size, 1])[:, 0]
            elif self.mode == 'general':
                self.w_att = tf.get_variable('w_att', shape=[dec_hid_size, enc_hid_size])

    def __call__(self, all_enc_hidden_sates, prev_state, inp_mask):
        """
        Computes attention response and weights
        Input shapes:
        all_enc_hidden_sates: [batch_size, time, enc_hid_size]
        prev_state: [batch_size, dec_hid_size]
        inp_mask: [batch_size, ninp]
        Output shapes:
        attn: [batch_size, enc_size]
        probs: [batch_size, ninp]
        """
        with tf.variable_scope(self.name):

            if self.mode == 'concat':
                print('last_decoded', prev_state)
                print(self.w_att)
                # pr = tf.tensordot(last_decoded, self.dec, axes=(1, 0))
                w_prod_s = tf.matmul(prev_state, self.w_att)  # [batch_size x hid_size]
                print('pr', w_prod_s)
                w_prod_s = tf.expand_dims(w_prod_s, 1)  # [dec_hid_size x 1 x hid_size]
                u_prod_all_enc_h = tf.tensordot(all_enc_hidden_sates, self.u_att, axes=(2, 0))  # [batch_size, time, hid_size]

                w_prod_s = tf.tile(w_prod_s, [1, tf.shape(u_prod_all_enc_h)[1], 1]) # [batch_size, time, hid_size]

                e = w_prod_s + u_prod_all_enc_h
                e = tf.tensordot(self.activ(e), self.vec, axes=(2, 0))  # [batch_size, time]

            elif self.mode == 'linear':

                prev_state_full = tf.expand_dims(prev_state, 1)     # [batch_size x 1 x dec_hid_size]
                prev_state_full = tf.tile(prev_state_full,
                                          [1, tf.shape(all_enc_hidden_sates)[1], 1])
                e = tf.reduce_sum(prev_state_full * all_enc_hidden_sates, axis=-1)

            else:
                w_prod_s = tf.matmul(prev_state, self.w_att)
                w_prod_s = tf.expand_dims(w_prod_s, 1)
                w_prod_s = tf.tile(w_prod_s, [1, tf.shape(all_enc_hidden_sates)[1], 1])
                e = tf.reduce_sum(w_prod_s * all_enc_hidden_sates, axis=-1)

            probs = tf.exp(e) * inp_mask

            probs = probs / tf.reduce_sum(probs, axis=-1, keep_dims=True)  # [batch_size, time]
            probs_full = tf.tile(tf.expand_dims(probs, 2), [1, 1, all_enc_hidden_sates.shape[2]])  # [batch_size, time, enc_hid_size]
            attn = tf.reduce_sum(probs_full * all_enc_hidden_sates, axis=1)  # [batch_size, enc_hid_size]

            return attn, probs


def initialize_uninitialized(sess = None):
    sess = sess or tf.get_default_session()
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def infer_length(seq, eos_idx, time_major=False, dtype=tf.int32):
    """
    compute length given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :returns: lengths, int32 vector of shape [batch]
    """
    axis = 0 if time_major else 1
    is_eos = tf.cast(tf.equal(seq, eos_idx), dtype)
    count_eos = tf.cumsum(is_eos, axis=axis, exclusive=True)
    lengths = tf.reduce_sum(tf.cast(tf.equal(count_eos, 0), dtype), axis=axis)
    return lengths


def infer_mask(seq, eos_idx, time_major=False, dtype=tf.float32):
    """
    compute mask given output indices and eos code
    :param seq: tf matrix [time,batch] if time_major else [batch,time]
    :param eos_ix: integer index of end-of-sentence token
    :returns: mask, float32 matrix with '0's and '1's of same shape as seq
    """
    axis = 0 if time_major else 1
    lengths = infer_length(seq, eos_idx, time_major=time_major)
    mask = tf.sequence_mask(lengths, maxlen=tf.shape(seq)[axis], dtype=dtype)
    if time_major: mask = tf.transpose(mask)
    return mask


def select_values_over_last_axis(values, indices):
    """
    Auxiliary function to select logits corresponding to chosen tokens.
    :param values: logits for all actions: float32[batch,tick,action]
    :param indices: action ids int32[batch,tick]
    :returns: values selected for the given actions: float[batch,tick]
    """
    assert values.shape.ndims == 3 and indices.shape.ndims == 2
    batch_size, seq_len = tf.shape(indices)[0], tf.shape(indices)[1]
    batch_i = tf.tile(tf.range(0, batch_size)[:, None], [1, seq_len])
    time_i = tf.tile(tf.range(0, seq_len)[None, :], [batch_size, 1])
    indices_nd = tf.stack([batch_i, time_i, indices], axis=-1)
    return tf.gather_nd(values, indices_nd)
