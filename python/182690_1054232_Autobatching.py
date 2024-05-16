get_ipython().system('pip install git+https://github.com/clab/dynet#egg=dynet')

import dynet as dy
import numpy as np

# acceptor LSTM
class LstmAcceptor(object):
    def __init__(self, in_dim, lstm_dim, out_dim, model):
        self.builder = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)
        self.W       = model.add_parameters((out_dim, lstm_dim))
    
    def __call__(self, sequence):
        lstm = self.builder.initial_state()
        W = self.W.expr() # convert the parameter into an Expession (add it to graph)
        outputs = lstm.transduce(sequence)
        result = W*outputs[-1]
        return result

# usage:
VOCAB_SIZE = 1000
EMBED_SIZE = 100

m = dy.Model()
trainer = dy.AdamTrainer(m)
embeds = m.add_lookup_parameters((VOCAB_SIZE, EMBED_SIZE))
acceptor = LstmAcceptor(EMBED_SIZE, 100, 3, m)


# training code
sum_of_losses = 0.0
for epoch in range(10):
    for sequence,label in [((1,4,5,1),1), ((42,1),2), ((56,2,17),1)]:
        dy.renew_cg() # new computation graph
        vecs = [embeds[i] for i in sequence]
        preds = acceptor(vecs)
        loss = dy.pickneglogsoftmax(preds, label)
        sum_of_losses += loss.npvalue()
        loss.backward()
        trainer.update()
    print sum_of_losses / 3
    sum_of_losses = 0.0
        
print "\n\nPrediction time!\n"
# prediction code:
for sequence in [(1,4,12,1), (42,2), (56,2,17)]:
    dy.renew_cg() # new computation graph
    vecs = [embeds[i] for i in sequence]
    preds = dy.softmax(acceptor(vecs))
    vals  = preds.npvalue()
    print np.argmax(vals), vals
    

    

class TreeRNN(object):
    def __init__(self, model, word_vocab, hdim):
        self.W = model.add_parameters((hdim, 2*hdim))
        self.E = model.add_lookup_parameters((len(word_vocab),hdim))
        self.w2i = word_vocab

    def __call__(self, tree): return self.expr_for_tree(tree)
    
    def expr_for_tree(self, tree):
        if tree.isleaf():
            return self.E[self.w2i.get(tree.label,0)]
        if len(tree.children) == 1:
            assert(tree.children[0].isleaf())
            expr = self.expr_for_tree(tree.children[0])
            return expr
        assert(len(tree.children) == 2),tree.children[0]
        e1 = self.expr_for_tree(tree.children[0], decorate)
        e2 = self.expr_for_tree(tree.children[1], decorate)
        W = dy.parameter(self.W)
        expr = dy.tanh(W*dy.concatenate([e1,e2]))
        return expr

# training code: batched.
for epoch in range(10):
    dy.renew_cg()     # we create a new computation graph for the epoch, not each item.
    # we will treat all these 3 datapoints as a single batch
    losses = []
    for sequence,label in [((1,4,5,1),1), ((42,1),2), ((56,2,17),1)]:
        vecs = [embeds[i] for i in sequence]
        preds = acceptor(vecs)
        loss = dy.pickneglogsoftmax(preds, label)
        losses.append(loss)
    # we accumulated the losses from all the batch.
    # Now we sum them, and do forward-backward as usual.
    # Things will run with efficient batch operations.
    batch_loss = dy.esum(losses)/3
    print batch_loss.npvalue() # this calls forward on the batch
    batch_loss.backward()
    trainer.update()
   
print "\n\nPrediction time!\n"
# prediction code:
dy.renew_cg() # new computation graph
batch_preds = []
for sequence in [(1,4,12,1), (42,2), (56,2,17)]:
    vecs = [embeds[i] for i in sequence]
    preds = dy.softmax(acceptor(vecs))
    batch_preds.append(preds)

# now that we accumulated the prediction expressions,
# we run forward on all of them:
dy.forward(batch_preds)
# and now we can efficiently access the individual values:
for preds in batch_preds:
    vals  = preds.npvalue()
    print np.argmax(vals), vals



