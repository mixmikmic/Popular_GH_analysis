import numpy as np
np.random.seed(2017)

import torch
torch.manual_seed(2017)

from scipy.misc import logsumexp # Use it for reference checking implementation

seq_length, num_states=4, 2
emissions = np.random.randint(20, size=(seq_length,num_states))*1.
transitions = np.random.randint(10, size=(num_states, num_states))*1.
print("Emissions:", emissions, sep="\n")
print("Transitions:", transitions, sep="\n")

def viterbi_decoding(emissions, transitions):
    # Use help from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/crf/python/ops/crf.py
    scores = np.zeros_like(emissions)
    back_pointers = np.zeros_like(emissions, dtype="int")
    scores = emissions[0]
    # Generate most likely scores and paths for each step in sequence
    for i in range(1, emissions.shape[0]):
        score_with_transition = np.expand_dims(scores, 1) + transitions
        scores = emissions[i] + score_with_transition.max(axis=0)
        back_pointers[i] = np.argmax(score_with_transition, 0)
    # Generate the most likely path
    viterbi = [np.argmax(scores)]
    for bp in reversed(back_pointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    viterbi_score = np.max(scores)
    return viterbi_score, viterbi

viterbi_decoding(emissions, transitions)

def viterbi_decoding_torch(emissions, transitions):
    scores = torch.zeros(emissions.size(1))
    back_pointers = torch.zeros(emissions.size()).int()
    scores = scores + emissions[0]
    # Generate most likely scores and paths for each step in sequence
    for i in range(1, emissions.size(0)):
        scores_with_transitions = scores.unsqueeze(1).expand_as(transitions) + transitions
        max_scores, back_pointers[i] = torch.max(scores_with_transitions, 0)
        scores = emissions[i] + max_scores
    # Generate the most likely path
    viterbi = [scores.numpy().argmax()]
    back_pointers = back_pointers.numpy()
    for bp in reversed(back_pointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    viterbi_score = scores.numpy().max()
    return viterbi_score, viterbi
    

viterbi_decoding_torch(torch.Tensor(emissions), torch.Tensor(transitions))

viterbi_decoding(emissions, transitions)

def log_sum_exp(vecs, axis=None, keepdims=False):
    ## Use help from: https://github.com/scipy/scipy/blob/v0.18.1/scipy/misc/common.py#L20-L140
    max_val = vecs.max(axis=axis, keepdims=True)
    vecs = vecs - max_val
    if not keepdims:
        max_val = max_val.squeeze(axis=axis)
    out_val = np.log(np.exp(vecs).sum(axis=axis, keepdims=keepdims))
    return max_val + out_val

def score_sequence(emissions, transitions, tags):
    # Use help from: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/crf/python/ops/crf.py
    score = emissions[0][tags[0]]
    for i, emission in enumerate(emissions[1:]):
        score = score + transitions[tags[i], tags[i+1]] + emission[tags[i+1]]
    return score

score_sequence(emissions, transitions, [1,1,0,0])

correct_seq = [0, 0, 1, 1]
[transitions[correct_seq[i],correct_seq[i+1]] for i in range(len(correct_seq) -1)]

sum([transitions[correct_seq[i], correct_seq[i+1]] for i in range(len(correct_seq) -1)])

viterbi_decoding(emissions, transitions)

score_sequence(emissions, transitions, [0, 0, 1, 1])

def score_sequence_torch(emissions, transitions, tags):
    score = emissions[0][tags[0]]
    for i, emission in enumerate(emissions[1:]):
        score = score + transitions[tags[i], tags[i+1]] + emission[tags[i+1]]
    return score

score_sequence_torch(torch.Tensor(emissions), torch.Tensor(transitions), [0, 0, 1, 1])

def get_all_tags(seq_length, num_labels):
    if seq_length == 0:
        yield []
        return
    for sequence in get_all_tags(seq_length-1, num_labels):
        #print(sequence, seq_length)
        for label in range(num_labels):
            yield [label] + sequence        
list(get_all_tags(4,2))

def get_all_tags_dp(seq_length, num_labels):
    prior_tags = [[]]
    for i in range(1, seq_length+1):
        new_tags = []
        for label in range(num_labels):
            for tags in prior_tags:
                new_tags.append([label] + tags)
        prior_tags = new_tags
    return new_tags
list(get_all_tags_dp(2,2))

def brute_force_score(emissions, transitions):
    # This is for ensuring the correctness of the dynamic programming method.
    # DO NOT run with very high values of number of labels or sequence lengths
    for tags in get_all_tags_dp(*emissions.shape):
        yield score_sequence(emissions, transitions, tags)

        
brute_force_sequence_scores = list(brute_force_score(emissions, transitions))
print(brute_force_sequence_scores)

max(brute_force_sequence_scores) # Best score calcuated using brute force

log_sum_exp(np.array(brute_force_sequence_scores)) # Partition function

def forward_algorithm_naive(emissions, transitions):
    scores = emissions[0]
    # Get the log sum exp score
    for i in range(1,emissions.shape[0]):
        print(scores)
        alphas_t = np.zeros_like(scores) # Forward vars at timestep t
        for j in range(emissions.shape[1]):
            emit_score = emissions[i,j]
            trans_score = transitions.T[j]
            next_tag_var = scores + trans_score
            alphas_t[j] = log_sum_exp(next_tag_var) + emit_score
        scores = alphas_t
    return log_sum_exp(scores)

forward_algorithm_naive(emissions, transitions)

def forward_algorithm_vec_check(emissions, transitions):
    # This is for checking the correctedness of log_sum_exp function compared to scipy
    scores = emissions[0]
    scores_naive = emissions[0]
    # Get the log sum exp score
    for i in range(1, emissions.shape[0]):
        print(scores, scores_naive)
        scores = emissions[i] + logsumexp(
            scores_naive + transitions.T,
            axis=1)
        scores_naive = emissions[i] + np.array([log_sum_exp(
            scores_naive + transitions.T[j]) for j in range(emissions.shape[1])])
    print(scores, scores_naive)
    return logsumexp(scores), log_sum_exp(scores_naive)

forward_algorithm_vec_check(emissions, transitions)

def forward_algorithm(emissions, transitions):
    scores = emissions[0]
    # Get the log sum exp score
    for i in range(1, emissions.shape[0]):
        scores = emissions[i] + log_sum_exp(
            scores + transitions.T,
            axis=1)
    return log_sum_exp(scores)

forward_algorithm(emissions, transitions)

tt = torch.Tensor(emissions)
tt_max, _ = tt.max(1)

tt_max.expand_as(tt)

tt.sum(0)

tt.squeeze(0)

tt.transpose(-1,-2)

tt.ndimension()

def log_sum_exp_torch(vecs, axis=None):
    ## Use help from: http://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#sphx-glr-beginner-nlp-advanced-tutorial-py
    if axis < 0:
        axis = vecs.ndimension()+axis
    max_val, _ = vecs.max(axis)
    vecs = vecs - max_val.expand_as(vecs)
    out_val = torch.log(torch.exp(vecs).sum(axis))
    #print(max_val, out_val)
    return max_val + out_val

def forward_algorithm_torch(emissions, transitions):
    scores = emissions[0]
    # Get the log sum exp score
    transitions = transitions.transpose(-1,-2)
    for i in range(1, emissions.size(0)):
        scores = emissions[i] + log_sum_exp_torch(
            scores.expand_as(transitions) + transitions,
            axis=1)
    return log_sum_exp_torch(scores, axis=-1)

forward_algorithm_torch(torch.Tensor(emissions), torch.Tensor(transitions))



