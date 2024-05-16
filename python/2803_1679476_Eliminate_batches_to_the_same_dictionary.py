def eliminate_batches_to_same_dictionary(old_batches_path, new_batches_path):
    """
    Eliminate all batches from one folder to the same dictionary (in alphabetical order). 

    Parametrs:
    ----------
    old_batches_path : folder containing all batches
    
    new_batches_path : folder that will contain all batches after function implementation
    """
    
    list_of_words = get_words_from_batches(batches_path)
    main_dictionary = list_to_word_index_dictionary(list_of_words)
    
    for batch_path in sorted(glob.glob(batches_path + "/*.batch")):
        batch = artm.messages.Batch()
        
        with open(batch_path, "rb") as f:
            batch.ParseFromString(f.read())
        
        new_batch = rewrite_batch_with_dictionary(batch, main_dictionary)
        
        batch_name = batch_path[batch_path.rfind('/'):]
        
        with open(new_batches_path + batch_name, 'wb') as fout:
            fout.write(new_batch.SerializeToString())
    
    return 0 
        
         
def get_words_from_batches(batches_path):
    """
    Get set of words from the all batches and making one big dictionary for all of them
    """
    set_of_words = set()
    
    for batch_path in sorted(glob.glob(batches_path + "/*.batch")):
        batch = artm.messages.Batch()
        
        with open(batch_path, "rb") as f:
            batch.ParseFromString(f.read())
        
        set_of_words = set_of_words.union(set(batch.token))
        
    return sorted(list(set_of_words))


def list_to_word_index_dictionary(list_of_words):
    """
    Transform list of unique elements to the dictionary of format {element:element index}
    """
    return dict(zip(list_of_words, xrange(0, len(list_of_words))))


def list_to_index_word_dictionary(list_of_words):
    """
    Transform list of unique elements to the dictionary of format {element index:element}
    """

    return dict(zip( xrange(0, len(list_of_words)), list_of_words))


def rewrite_batch_with_dictionary(batch, main_dictionary):
    """
    Create new batch with the same content as the old batch, but with 
    tokens corresponds to tokens from main_dictionary
    
    Parametrs:
    ----------
    batch : old batch
    
    main_dictionary: element:element index dictionary of all collection
    """
    
    new_batch = artm.messages.Batch()
    new_batch.id = str(uuid.uuid4())
    
    for token in sorted(main_dictionary.keys()):
        new_batch.token.append(token)
        new_batch.class_id.append(u'@default_class')
    
    batch_dictionary = list_to_index_word_dictionary(batch.token)
    
    for old_item in batch.item:
        new_item = new_batch.item.add()
        new_item.id = old_item.id
        new_item.title = old_item.title

        for one_token_id, one_token_weight in zip(old_item.token_id, old_item.token_weight):
            new_item.token_id.append(main_dictionary[batch_dictionary[one_token_id]])
            new_item.token_weight.append(one_token_weight)    
    
    return new_batch

eliminate_batches_to_same_dictionary('batches/my_batches', 'batches/my_batches_new')

