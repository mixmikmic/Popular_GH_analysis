# First, we'll import some libraries of Python functions that we'll need to run this code
import pandas as pd
import sqlite3
import xlrd as xl

# Select an excel file saved in the same folder as this SQL_Practice file. 
path = ('book_db.xlsx')

# if this .sqlite db doesn't already exists, this will create it
# if the .sqlite db *does* already exist, this establishes the desired connection
con = sqlite3.connect("book_db.sqlite")

# this pulls out the names of the sheets in the workbook. We'll use these to name the different tables in the SQLite database that we'll create
table_names = xl.open_workbook(path).sheet_names()

# this loop makes it possible to use any other .xls sheet, since the sheet names aren't called out specifically
for table in table_names:
    df = pd.read_excel(path, sheetname='{}'.format(table))
    con.execute("DROP TABLE IF EXISTS {}".format(table))
    pd.io.sql.to_sql(df, "{}".format(table), con, index=False)

# now the spreadsheets are in tables in a mini database!

# Finally, a little function to make it easy to run queries on this mini-database
def run(query):
    results = pd.read_sql("{}".format(query), con)
    return results

run('''
PRAGMA TABLE_INFO(transactions)
''')

run('''
SELECT  
    author,
    COUNT(DISTINCT(title)) as unique_titles,
    SUM(CASE WHEN gender = 'F' THEN price*purchases end) AS female_revenue,
    SUM(CASE WHEN gender = 'M' THEN price*purchases end) AS male_revenue
FROM
    books B
    JOIN transactions T ON B.id = T.bookid
    JOIN users U on U.id = T.userID
    JOIN authors A on A.id = B.AuthorID
GROUP BY author
ORDER BY female_revenue + male_revenue DESC
LIMIT 10
    ''')

run('''
SELECT 
    title, 
    author,
    SUM(clicks)/COUNT(*) as CTR,
    SUM(spend)/SUM(clicks) as CPC,
    SUM(price*purchases)/SUM(clicks) as RPC,
    SUM(purchases) as Conversions, 
    SUM(purchases)/SUM(clicks) as Conversion_Rate,
    SUM(spend)/SUM(purchases) as COS,
    SUM(price*purchases)/COUNT(*)*1000 RPM
FROM
    books B
    JOIN transactions T ON B.id = T.bookid
    JOIN users U on U.id = T.userID
    JOIN authors A on A.id = B.AuthorID
GROUP BY title
ORDER BY RPM DESC
LIMIT 10
    ''')



