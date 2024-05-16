from pymongo import MongoClient as MCli



class IO_Mongo(object):
    """Connect to the mongo server on localhost at port 27017."""
    conn={'host':'localhost', 'ip':'27017'}


    # Initialize the class with client connection, the database (i.e. twtr_db), and the collection (i.e. twtr_coll)
    def __init__(self, db='twtr_db', coll='twtr_coll', **conn):
        """Connect to the MonfoDB server"""
        self.client = MCli(**conn)
        self.db = self.client[db]
        self.coll = self.db[coll]


    # The `save` method inserts new records in the pre_initialized collection and database
    def save(self, data):
        """ Insert data to collection in db. """
        return self.coll.insert(data)
    
    
    # The `load` method allows the retrieval of specific records
    def load(self, return_cursor=False, criteria=None, projection=None):
        """ The `load` method allows the retrieval of specific records according to criteria and projection. 
            In case of large amount of data, it returns a cursor.
        """
        if criteria is None:
            criteria = {}
        
        # Find record according to some criteria.
        if projection is None:
            cursor = self.coll.find(criteria)
        else:
            cursor = self.coll.find(criteria, projection)
        
        # Return a cursor for large amount of data
        if return_cursor:
            return cursor
        else:
            return [item for item in cursor]
        
        

f = IO_Mongo()
f.load()



