import pixiedust
pixiedust.enableJobMonitor()

# @hidden_cell
# Enter your DashDB JDBC URL (e.g. 'jdbc:db2://dashdb-entry-yp-dal00-00.services.dal.bluemix.net:50000/BLUDB')
jdbcurl = 'jdbc:db2://...'
# Enter your DashDB user name (e.g. 'dash0815')
user = '...'
# Enter your DashDB password (e.g. 'myvoiceismypassword')
password = '...'
# Enter your source table or view name (e.g. 'mytable')
table = '...'

# no changes are required to this cell
# obtain Spark SQL Context
sqlContext = SQLContext(sc)
# load data
props = {}
props['user'] = user
props['password'] = password
dashdb_data = sqlContext.read.jdbc(jdbcurl, table, properties=props)

display(dashdb_data)

