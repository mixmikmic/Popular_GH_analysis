import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

import time
from datetime import datetime
from IBWrapper import IBWrapper, contract
from ib.ext.EClientSocket import EClientSocket
from ib.ext.ScannerSubscription import ScannerSubscription

def obtain_ib_acct():
    return pd.read_excel("G:/live/CURRENT_BATCH_IB.xlsx", sheetname="2016S1R2")[['ADM_NO','NAME','IB ACCT']]

df = obtain_ib_acct()         # Obtain current batch IB account number        

callback = IBWrapper()        # Instantiate IBWrapper. callback 
tws = EClientSocket(callback) # Instantiate EClientSocket and return data to callback

host = ""
port = 4001
clientId = 202

tws.eConnect(host, port, clientId) # Connect to TWS

tws.setServerLogLevel(5)           # Set error output to verbose

create = contract()                # Instantiate contract class
callback.initiate_variables()

reqID = 1001
tws.reqAccountSummary(reqID,"All","NetLiquidation")

res = pd.DataFrame(callback.account_Summary,
                  columns = ["reqID", "IB ACCT", "Key", 
                             "NLV", "Currency"])[["IB ACCT", "NLV"]]
res["IB ACCT"].drop_duplicates(keep = "last", inplace=True)

current_batch_nlv = pd.merge(df, res, on="IB ACCT")

import matplotlib
get_ipython().magic('matplotlib inline')
matplotlib.style.use("ggplot")

Init_bal = 1000000
current_batch_nlv["PnL"] = current_batch_nlv["NLV"].astype(float) - Init_bal
current_batch_nlv.sort_values(by = "PnL", ascending=False, inplace=True)
current_batch_nlv.plot(kind="bar", x=current_batch_nlv["NAME"].str.split(" ").str.get(0))

tws.eDisconnect()

