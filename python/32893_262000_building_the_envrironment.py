import zipfile
s_fname = "data/petr4_0725_0818.zip"
archive = zipfile.ZipFile(s_fname, 'r')
def foo():
    f_total = 0.
    f_tot_rows = 0.
    for i, x in enumerate(archive.infolist()):
        f_total += x.file_size/ 1024.**2
        for num_rows, row in enumerate(archive.open(x)):
            f_tot_rows += 1
        print "{}:\t{:,.0f} rows\t{:0.2f} MB".format(x.filename, num_rows + 1, x.file_size/ 1024.**2)
    print '=' * 42
    print "TOTAL\t\t{} files\t{:0.2f} MB".format(i+1,f_total)
    print "\t\t{:0,.0f} rows".format(f_tot_rows)

get_ipython().magic('time foo()')

import pandas as pd
df = pd.read_csv(archive.open(x), index_col=0, parse_dates=['Date'])
df.head()

print "{:%m/%d/%Y}".format(df.Date[0])
print df.groupby('Type').count()['Date']

from bintrees import FastRBTree

def foo():
    for idx, row in df.iterrows():
        pass

print "time to iterate the rows:"
get_ipython().magic('time foo()')

def foo():
    bid_tree = FastRBTree()
    ask_tree = FastRBTree()
    for idx, row in df.iterrows():
        if row.Type == 'BID':
            bid_tree.insert(row['Price'], row['Size'])
        elif row.Type == 'ASK':
            ask_tree.insert(row['Price'], row['Size'])

print "time to insert everything in binary trees:"
get_ipython().magic('time foo()')

df_aux = df[df.Type == 'TRADE'].Price
df_aux.index = df[df.Type == 'TRADE'].Date
ax = df_aux.plot()
ax.set_title("Price fluctuation of PETR4\n");

import pandas as pd

df_all = None

for i, x in enumerate(archive.infolist()):
    df = pd.read_csv(archive.open(x), index_col=0, parse_dates=['Date'])
    ts_date = df.Date[0].date()
    df.Date = ["{:%H:%M:%S}".format(x) for x in df.Date]
    df = df[df.Type == "TRADE"]
    df = pd.DataFrame(df.groupby('Date').last()['Price'])
    if i == 0:
        df_all = df.copy()
        df_all.columns = [ts_date] 
    else:
        df_aux = df.copy()
        df_aux.columns = [ts_date]
        df_all = df_all.join(df_aux)
df_all.index = pd.to_datetime(df_all.index)
df_all = df_all.fillna(method='ffill')
df_all = df_all.dropna()

import numpy as np
df_logrtn = np.log(df_all/df_all.shift())
df_logrtn = df_logrtn[[(x.hour*60 + x.minute) < (16*60 + 55) for x in df_logrtn.index]]
ax = df_logrtn.cumsum().plot(legend=False)
ax.set_title('Cumulative Log-Returns of PETR4 in 19 different days\n', fontsize=16)
ax.set_ylabel('Return');

import qtrader.preprocess as preprocess
s_fname = "data/petr4_0725_0818.zip"
preprocess.make_zip_file(s_fname)

import zipfile
import pandas as pd
s_fname = "data/petr4_0725_0818.zip"
archive = zipfile.ZipFile(s_fname, 'r')


f_total = 0.
i_same = 0
i_different = 0
for i, x in enumerate(archive.infolist()):
    s_fname = 'data/petr4_0725_0818_2/' + x.filename
    df = pd.read_csv(archive.open(x), index_col=0, parse_dates=['Date'])
    df2 = pd.read_csv(s_fname, index_col=0, parse_dates=['Date'])
    f_all = (df.ix[df.Type=='TRADE', 'Price'] * df.ix[df.Type=='TRADE', 'Size']).sum()
    f_all2 = (df2.ix[df2.Type=='TRADE', 'Price'] * df2.ix[df2.Type=='TRADE', 'Size']).sum()
    if f_all == f_all2:
        i_same += 1
    else:
        i_different += 1

print "{} files has the same number of trades".format(i_same)
print "{} files has DIFFERENT number of trades".format(i_different)
    

# example of message
d_msg = {'instrumento_symbol': 'PETR4',
         'agent_id': 10,
         'order_entry_step': 15,
         'order_status': 'New',
         'last_order_id': 0,
         'order_id': 0,
         'order_side': 'BID',
         'order_price': 12.11,
         'total_qty_order': 100,
         'traded_qty_order': 0}

class Order(object):
    '''
    A representation of a single Order
    '''
    def __init__(self, d_msg):
        '''
        Instantiate a Order object. Save all parameter as attributes
        :param d_msg: dictionary.
        '''
        # keep data extract from file
        self.d_msg = d_msg.copy()
        self.d_msg['org_total_qty_order'] = self.d_msg['total_qty_order']
        f_q1 = self.d_msg['total_qty_order']
        f_q2 = self.d_msg['traded_qty_order']
        self.d_msg['total_qty_order'] = f_q1 - f_q2
        self.order_id = d_msg['order_id']
        self.last_order_id = d_msg['last_order_id']
        self.name = "{:07d}".format(d_msg['order_id'])
        self.main_id = self.order_id

    def __str__(self):
        '''
        Return the name of the Order
        '''
        return self.name

    def __repr__(self):
        '''
        Return the name of the Order
        '''
        return self.name

    def __eq__(self, other):
        '''
        Return if a Order has equal order_id from the other
        :param other: Order object. Order to be compared
        '''
        return self.order_id == other.order_id

    def __ne__(self, other):
        '''
        Return if a Order has different order_id from the other
        :param other: Order object. Order to be compared
        '''
        return not self.__eq__(other)

    def __hash__(self):
        '''
        Allow the Order object be used as a key in a hash table. It is used by
        dictionaries
        '''
        return self.order_id.__hash__()

    def __getitem__(self, s_key):
        '''
        Allow direct access to the inner dictionary of the object
        :param i_index: integer. index of the l_legs attribute list
        '''
        return self.d_msg[s_key]

my_order = Order(d_msg)
print "My id is {} and the price is {:0.2f}".format(my_order['order_id'], my_order['order_price'])
print "The string representation of the order is {}".format(my_order)

from bintrees import FastRBTree

class PriceLevel(object):
    '''
    A representation of a Price level in the book
    '''
    def __init__(self, f_price):
        '''
        A representation of a PriceLevel object
        '''
        self.f_price = f_price
        self.i_qty = 0
        self.order_tree = FastRBTree()

    def add(self, order_aux):
        '''
        Insert the information in the tree using the info in order_aux. Return
        is should delete the Price level or not
        :param order_aux: Order Object. The Order message to be updated
        '''
        # check if the order_aux price is the same of the self
        if order_aux['order_price'] != self.f_price:
            raise DifferentPriceException
        elif order_aux['order_status'] == 'limit':
            self.order_tree.insert(order_aux.main_id, order_aux)
            self.i_qty += int(order_aux['total_qty_order'])
        # check if there is no object in the updated tree (should be deleted)
        return self.order_tree.count == 0

    def delete(self, i_last_id, i_old_qty):
        '''
        Delete the information in the tree using the info in order_aux. Return
        is should delete the Price level or not
        :param i_last_id: Integer. The previous secondary order id
        :param i_old_qty: Integer. The previous order qty
        '''
        # check if the order_aux price is the same of the self
        try:
            self.order_tree.remove(i_last_id)
            self.i_qty -= i_old_qty
        except KeyError:
            raise DifferentPriceException
        # check if there is no object in the updated tree (should be deleted)
        return self.order_tree.count == 0

    def __str__(self):
        '''
        Return the name of the PriceLevel
        '''
        return '{:,.0f}'.format(self.i_qty)

    def __repr__(self):
        '''
        Return the name of the PriceLevel
        '''
        return '{:,.0f}'.format(self.i_qty)

    def __eq__(self, other):
        '''
        Return if a PriceLevel has equal price from the other
        :param other: PriceLevel object. PriceLevel to be compared
        '''
        # just to make sure that there is no floating point discrepance
        f_aux = other
        if not isinstance(other, float):
            f_aux = other.f_price
        return abs(self.f_price - f_aux) < 1e-4

    def __gt__(self, other):
        '''
        Return if a PriceLevel has a gerater price from the other.
        Bintrees uses that to compare nodes
        :param other: PriceLevel object. PriceLevel to be compared
        '''
        # just to make sure that there is no floating point discrepance
        f_aux = other
        if not isinstance(other, float):
            f_aux = other.f_price
        return (f_aux - self.f_price) > 1e-4

    def __lt__(self, other):
        '''
        Return if a Order has smaller order_id from the other. Bintrees uses
        that to compare nodes
        :param other: Order object. Order to be compared
        '''
        f_aux = other
        if not isinstance(other, float):
            f_aux = other.f_price
        return (f_aux - self.f_price) < -1e-4

    def __ne__(self, other):
        '''
        Return if a Order has different order_id from the other
        :param other: Order object. Order to be compared
        '''
        return not self.__eq__(other)

my_order = Order(d_msg)

# create different orders at the same price
d_msg1 = d_msg.copy()
d_msg1['order_id'] = 1
order1 = Order(d_msg1)
d_msg2 = d_msg.copy()
d_msg2['order_id'] = 2
order2 = Order(d_msg2)
d_msg3 = d_msg.copy()
d_msg3['order_id'] = 3
order3 = Order(d_msg3)

my_price = PriceLevel(d_msg['order_price'])
my_price.add(order1)
my_price.add(order2)
my_price.add(order3)

print "There is {} shares at {:.2f}".format(my_price, my_price.f_price)
print 'the orders in the book are: {}'.format(dict(my_price.order_tree))

my_price.delete(1, 100)
my_price.delete(2, 100)
print "There is {} shares at {:.2f}".format(my_price, my_price.f_price)
print 'the orders in the book are: {}'.format(dict(my_price.order_tree))

import qtrader.book as book; reload(book);

my_book = book.LimitOrderBook('PETR4')

d_msg0 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 0,
          'order_price': 12.12,
          'order_side': 'ASK',
          'order_status': 'New',
          'total_qty_order': 400,
          'traded_qty_order': 0,
          'agressor_indicator': 'Neutral'}


d_msg1 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 1,
          'order_price': 12.11,
          'order_side': 'BID',
          'order_status': 'New',
          'total_qty_order': 100,
          'traded_qty_order': 0,
          'agressor_indicator': 'Neutral'}

d_msg2 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 2,
          'order_price': 12.11,
          'order_side': 'BID',
          'order_status': 'New',
          'total_qty_order': 100,
          'traded_qty_order': 0,
          'agressor_indicator': 'Neutral'}

d_msg3 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 3,
          'order_price': 12.10,
          'order_side': 'BID',
          'order_status': 'New',
          'total_qty_order': 200,
          'traded_qty_order': 0,
          'agressor_indicator': 'Neutral'}

d_msg4 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 4,
          'order_price': 12.10,
          'order_side': 'BID',
          'order_status': 'New',
          'total_qty_order': 100,
          'traded_qty_order': 0,
          'agressor_indicator': 'Neutral'}

d_msg5 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 3,
          'order_price': 12.10,
          'order_side': 'BID',
          'order_status': 'Replaced',
          'total_qty_order': 100,
          'traded_qty_order': 0,
          'agressor_indicator': 'Neutral'}

d_msg6 = {'agent_id': 10,
          'instrumento_symbol': 'PETR4',
          'last_order_id': 0,
          'order_entry_step': 15,
          'order_id': 1,
          'order_price': 12.11,
          'order_side': 'BID',
          'order_status': 'Filled',
          'total_qty_order': 100,
          'traded_qty_order': 0,
          'agressor_indicator': 'Passive'}

# include several orders
my_book.update(d_msg0)
my_book.update(d_msg1)
my_book.update(d_msg2)
my_book.update(d_msg3)
my_book.update(d_msg4)
my_book.get_n_top_prices(5)

# test cancelation
my_book.update(d_msg5)
my_book.get_n_top_prices(5)

# checking if the order of Ids are OK
x = my_book.book_bid.price_tree.get(12.10)
x.order_tree

# test a trade
my_book.update(d_msg6)
my_book.get_n_top_prices(5)

my_book.get_basic_stats()

import zipfile
s_fname = "data/petr4_0725_0818.zip"
archive = zipfile.ZipFile(s_fname, 'r')

f_total = 0.
for i, x in enumerate(archive.infolist()):
    f_total += x.file_size/ 1024.**2
    for num_rows, row in enumerate(archive.open(x)):
        pass

import pandas as pd
df = pd.read_csv(archive.open(x), index_col=0, parse_dates=['Date'])
df.head(5)

def translate_row(idx, row, i_order_id):
    '''
    '''
    if row.Type != 'TRADE' and row['Size'] > 100:
        d_rtn = {'agent_id': 10,
                 'instrumento_symbol': 'PETR4',
                 'new_order_id': i_order_id + 1,
                 'order_entry_step': idx,
                 'order_id': i_order_id + 1,
                 'order_price': row['Price'],
                 'order_side': row.Type,
                 'order_status': 'New',
                 'total_qty_order': row['Size'],
                 'traded_qty_order': 0,
                 'agressor_indicator': 'Neutral'}
        return i_order_id + 1,  d_rtn

# test the structure
import qtrader.book as book; reload(book);
my_book = book.LimitOrderBook('PETR4')
for idx, row in df.iterrows():
    i_id = my_book.i_last_order_id
    t_rtn = translate_row(idx, row, i_id)
    if t_rtn:
        my_book.i_last_order_id = t_rtn[0]
        my_book.update(t_rtn[1])
    if idx == 1000:
        break

my_book.get_n_top_prices(5)

my_book.get_basic_stats()

def translate_row(idx, row, my_book):
    '''
    '''
    l_msg = []
    if row.Type != 'TRADE' and row['Size'] % 100 == 0:
        # recover the best price
        f_best_price = my_book.get_best_price(row.Type)
        i_order_id = my_book.i_last_order_id + 1
        # check if there is orders in the row price
        obj_ordtree = my_book.get_orders_by_price(row.Type, row['Price'])
        if obj_ordtree:
            # cant present more than 2 orders (mine and market)
            assert len(obj_ordtree) <= 2, 'More than two offers'
            # get the first order
            obj_order = obj_ordtree.nsmallest(1)[0][1]
            # check if should cancel the best price
            b_cancel = False
            if row.Type == 'BID' and row['Price'] < f_best_price:
                # check if the price in the row in smaller
                obj_ordtree2 = my_book.get_orders_by_price(row.Type)
                best_order = obj_ordtree2.nsmallest(1)[0][1]
                d_rtn = best_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                l_msg.append(d_rtn.copy())
            elif row.Type == 'ASK' and row['Price'] > f_best_price:
                obj_ordtree2 = my_book.get_orders_by_price(row.Type)
                best_order = obj_ordtree2.nsmallest(1)[0][1]
                d_rtn = best_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                l_msg.append(d_rtn.copy())

            
            
            # replace the current order
            i_old_id = obj_order.main_id
            i_new_id = obj_order.main_id
            if row['Size'] > obj_order['total_qty_order']:
                i_new_id = my_book.i_last_order_id + 1
                d_rtn = obj_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                l_msg.append(d_rtn.copy())


            # Replace the order
            d_rtn = {'agent_id': 10,
                     'instrumento_symbol': 'PETR4',
                     'order_id': i_old_id,
                     'order_entry_step': idx,
                     'new_order_id': i_new_id,
                     'order_price': row['Price'],
                     'order_side': row.Type,
                     'order_status': 'Replaced',
                     'total_qty_order': row['Size'],
                     'traded_qty_order': 0,
                     'agressor_indicator': 'Neutral'}
            l_msg.append(d_rtn.copy())
        else:               
            # if the price is not still in the book, include a new order
            d_rtn = {'agent_id': 10,
                     'instrumento_symbol': 'PETR4',
                     'order_id': my_book.i_last_order_id + 1,
                     'order_entry_step': idx,
                     'new_order_id': my_book.i_last_order_id + 1,
                     'order_price': row['Price'],
                     'order_side': row.Type,
                     'order_status': 'New',
                     'total_qty_order': row['Size'],
                     'traded_qty_order': 0,
                     'agressor_indicator': 'Neutral'}
            l_msg.append(d_rtn)
        return l_msg

# test the structure
import qtrader.book as book; reload(book);
import pprint
import time
f_start = time.time()
my_book = book.LimitOrderBook('PETR4')
self = my_book
for idx, row in df.iterrows():
    l_msg = translate_row(idx, row, my_book)
    if l_msg:
        for msg in l_msg:
            my_book.update(msg)
#     if idx == 10000:
#         break
"It took {:0.2f} seconds to process {:0,.0f} rows".format(time.time() - f_start, idx + 1)

my_book.get_basic_stats()

my_book.get_n_top_prices(100)

df.head()

def foo():
    best_bid = None
    best_ask = None
    i_num_cross = 0
    l_cross = []
    for idx, row in df.iterrows():
        if row.Size % 100 == 0:
            if row.Type == 'BID':
                best_bid = row.copy()
            elif row.Type == 'ASK':
                best_ask = row.copy()
            else:
                if not isinstance(best_bid, type(None)):
                    if row.Price == best_bid.Price:
                        if row.Size > best_bid.Size:
    #                         print 'cross-bid', idx
                            i_num_cross += 1
                            l_cross.append(idx)

                if not isinstance(best_ask, type(None)):
                    if row.Price == best_ask.Price:
                        if row.Size > best_ask.Size:
    #                         print 'cross-ask', idx
                            i_num_cross += 1
                            l_cross.append(idx)
    print "number of cross-orders: {:.0f}".format(i_num_cross)
    return l_cross

get_ipython().magic('time l_cross = foo()')

import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.book as book; reload(book);
import time
s_fname = "data/petr4_0725_0818.zip"
my_test = matching_engine.BloombergMatching(None, "PETR4", 200, s_fname)
f_start = time.time()
for i in xrange(my_test.max_nfiles):
    for d_data in my_test:
        pass
print "Time to iterate the files: {:0.2f}".format(time.time() - f_start)

import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.book as book; reload(book);
import time
s_fname = "data/petr4_0725_0818_2.zip"
my_test = matching_engine.BloombergMatching(None, "PETR4", 200, s_fname)
f_start = time.time()

for i in xrange(my_test.max_nfiles):
    quit = False
    while True:
        try:
            l_msg = my_test.next()
        except StopIteration:
            quit = True
        finally:
            if quit:
                break
#     break
print "Time to iterate the files: {:0.2f}".format(time.time() - f_start)

my_test.my_book.get_n_top_prices(5)

import qtrader.book as book; reload(book);
import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.environment as environment;  reload(environment);
import pprint

e = environment.Environment()
a = e.create_agent(environment.Agent)
e.set_primary_agent(a)
pprint.pprint(dict(e.agent_states))

import qtrader.book as book; reload(book);
import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.environment as environment;  reload(environment);
import pprint
import time
s_fname = "data/petr4_0725_0818_2.zip"
my_env = environment.Environment()
f_start = time.time()

for i in xrange(my_env.order_matching.max_nfiles):
    quit = False
    my_env.reset()
    while True:
        try:
            l_msg = my_env.step()
        except StopIteration:
            quit = True
        finally:
            if quit:
                break

print "Time to iterate the files: {:0.2f}".format(time.time() - f_start)

import qtrader.book as book; reload(book);
import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.environment as environment;  reload(environment);
import pprint
import time
s_fname = "data/petr4_0725_0818_2.zip"
my_env = environment.Environment()
f_start = time.time()

for i in xrange(my_env.order_matching.max_nfiles):
    quit = False
    my_env.reset()
    while True:
        try:
#             if my_env.order_matching.idx == 6:
#                 if my_env.order_matching.i_nrow == 107410:
#                     raise NotImplementedError
            if my_env.order_matching.i_nrow > 9:
                break
            l_msg = my_env.step()
        except StopIteration:
            quit = True
        finally:
            if quit:
                break
    break
print "Time to iterate the files: {:0.2f}".format(time.time() - f_start)

import qtrader.book as book; reload(book);
import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.environment as environment;  reload(environment);
import pprint
import time
s_fname = "data/petr4_0725_0818_2.zip"
my_env = environment.Environment()
f_start = time.time()

for i in xrange(my_env.order_matching.max_nfiles):
    quit = False
    my_env.reset()
    while True:
        try:
#             if my_env.order_matching.idx == 6:
#                 if my_env.order_matching.i_nrow == 107410:
#                     raise NotImplementedError
            if my_env.order_matching.i_nrow > 9:
                break
            l_msg = my_env.step()
        except StopIteration:
            quit = True
        finally:
            if quit:
                break
    break
print "Time to iterate the files: {:0.2f}".format(time.time() - f_start)

import qtrader.book as book; reload(book);
import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.environment as environment;  reload(environment);
import qtrader.agent as agent; reload(agent);
import qtrader.simulator as simulator; reload(simulator);

e = environment.Environment()
a = e.create_agent(agent.BasicAgent)
e.set_primary_agent(a)

sim = simulator.Simulator(e)
get_ipython().magic('time sim.run(n_trials=20)')

import qtrader.book as book; reload(book);
import qtrader.matching_engine as matching_engine; reload(matching_engine);
import qtrader.environment as environment;  reload(environment);
import qtrader.agent as agent; reload(agent);
import qtrader.simulator as simulator; reload(simulator);
import qtrader.translators as translators; reload(translators);

e = environment.Environment()
a = e.create_agent(agent.BasicAgent, f_min_time=3600.)
e.set_primary_agent(a)

sim = simulator.Simulator(e)
get_ipython().magic('time sim.run(n_trials=30)')



#loading style sheet
from IPython.core.display import HTML
HTML( open('ipython_style.css').read())

#changing matplotlib defaults
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})
sns.set_style("whitegrid")
sns.set_palette(sns.color_palette("Set2", 10))



