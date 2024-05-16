import swat

conn = swat.CAS(host, port, username, password)

conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/class.csv', 
              casout=dict(name='class', caslib='casuser'))

conn.tableinfo(caslib='casuser')

cls = conn.CASTable('class', caslib='CASUSER')
cls

cls.to_frame()

conn.close()



