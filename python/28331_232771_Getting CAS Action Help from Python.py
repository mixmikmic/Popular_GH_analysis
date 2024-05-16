import swat

conn = swat.CAS(host, port, username, password)

out = conn.help()

out = conn.help(actionset='simple')

out = conn.help(action='summary')

help(conn.simple)

help(conn.simple.summary)

get_ipython().magic('pinfo conn.simple')

get_ipython().magic('pinfo conn.simple.summary')

conn.close()



