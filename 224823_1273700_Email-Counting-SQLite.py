import sqlite3
conn = sqlite3.connect('emaildb.sqlite')
cur=conn.cursor()

cur.execute('DROP TABLE IF EXISTS Counts')
cur.execute('''
CREATE TABLE Counts (email TEXT, count INTEGER)''')

import urllib.request, urllib.parse, urllib.error

print("Opening the file connection...")
# Following example reads Project Gutenberg EBook of Pride and Prejudice
fhand = urllib.request.urlopen('http://data.pr4e.org/mbox.txt')

txt_dump = ''
line_count=0
word_count=0
# Iterate over the lines in the file handler object and dump the data into the text string. 
# Also increment line and word counts
for line in fhand:
# Use decode method to convert the UTF-8 to Unicode string
    txt_dump+=line.decode()
    line_count+=1
    # Count the length of words in the line and add to the running count
    word_count+=len(line.decode().split(' '))

# Prints basic informationn about the text data
print("Printing some info on the text dump\n"+"-"*60)
print("Total characters:",len(txt_dump))
print("Total words:",word_count)
print(f"Total lines: {line_count}")

file = open('mbox.txt','w') 
file.write(txt_dump)
file.close()

fh=open('mbox.txt')

show_text=fh.read(1000)
print(show_text)

for line in fh:
    if not line.startswith('From: '): continue
    pieces = line.split()
    email = pieces[1]
    cur.execute('SELECT count FROM Counts WHERE email = ? ', (email,))
    row = cur.fetchone()
    if row is None:
        cur.execute('''INSERT INTO Counts (email, count)
                VALUES (?, 1)''', (email,))
    else:
        cur.execute('UPDATE Counts SET count = count + 1 WHERE email = ?',
                    (email,))
conn.commit()

sqlstr = 'SELECT email,count FROM Counts ORDER BY count DESC LIMIT 20'

for row in cur.execute(sqlstr):
    print(str(row[0]), row[1])

sqlstr = 'SELECT AVG(count) FROM Counts WHERE email LIKE "%umich%"'
for row in cur.execute(sqlstr):
    print(float(row[0]))

fh.close()
cur.close()
fhand.close()

