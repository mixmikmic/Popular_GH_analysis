import swat

conn = swat.CAS(host, port, username, password)

tbl = conn.read_csv('https://raw.githubusercontent.com/sassoftware/sas-viya-programming/master/data/cars.csv')
tbl

tbl.head()

print(tbl.to_csv())

print(tbl.to_html())

print(tbl.to_latex())

conn.close()



