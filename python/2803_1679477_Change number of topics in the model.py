import artm
print artm.version()

batch_vectorizer = artm.BatchVectorizer(data_path=r'C:\bigartm\data', data_format='bow_uci', collection_name='kos')
model = artm.ARTM(topic_names=['topic_{}'.format(i) for i in xrange(5)],
                  scores=[artm.PerplexityScore(name='PerplexityScore')],
                  num_document_passes = 10)
model.initialize(dictionary=batch_vectorizer.dictionary)
model.fit_offline(batch_vectorizer=batch_vectorizer, num_collection_passes=5)
print model.score_tracker['PerplexityScore'].value

model.master.merge_model({'nwt': 1.0}, 'test', topic_names = ['topic_{}'.format(i) for i in xrange(7)])
model.get_phi(model_name='test')[:5]

for model_description in model.info.model:
    print model_description

(test_model, test_matrix) = model.master.attach_model('test')
for model_description in model.info.model:
    print model_description

import numpy as np
test_matrix[:, [5,6]] = np.random.rand(test_matrix.shape[0], 2)
model.get_phi(model_name='test')[:5]

# Fitting model with our internal API --- process batches and normalize model
model.initialize(dictionary=batch_vectorizer.dictionary)
for i in xrange(5):
    model.master.clear_score_cache()
    model.master.process_batches(model._model_pwt, model._model_nwt,
                                 batches=[x.filename for x in batch_vectorizer.batches_list],
                                 num_document_passes = 10)
    model.master.normalize_model(model._model_pwt, model._model_nwt)
    print model.get_score('PerplexityScore').value

