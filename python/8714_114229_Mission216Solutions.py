import sqlite3
conn = sqlite3.connect("nominations.db")
schema = conn.execute("pragma table_info(nominations);").fetchall()
first_ten = conn.execute("select * from nominations limit 10;").fetchall()

for r in schema:
    print(r)
    
for r in first_ten:
    print(r)

years_hosts = [(2010, "Steve Martin"),
               (2009, "Hugh Jackman"),
               (2008, "Jon Stewart"),
               (2007, "Ellen DeGeneres"),
               (2006, "Jon Stewart"),
               (2005, "Chris Rock"),
               (2004, "Billy Crystal"),
               (2003, "Steve Martin"),
               (2002, "Whoopi Goldberg"),
               (2001, "Steve Martin"),
               (2000, "Billy Crystal"),
            ]
create_ceremonies = "create table ceremonies (id integer primary key, year integer, host text);"
conn.execute(create_ceremonies)
insert_query = "insert into ceremonies (Year, Host) values (?,?);"
conn.executemany(insert_query, years_hosts)

print(conn.execute("select * from ceremonies limit 10;").fetchall())
print(conn.execute("pragma table_info(ceremonies);").fetchall())

conn.execute("PRAGMA foreign_keys = ON;")

create_nominations_two = '''create table nominations_two 
(id integer primary key, 
category text, 
nominee text, 
movie text, 
character text, 
won text,
ceremony_id integer,
foreign key(ceremony_id) references ceremonies(id));
'''

nom_query = '''
select ceremonies.id as ceremony_id, nominations.category as category, 
nominations.nominee as nominee, nominations.movie as movie, 
nominations.character as character, nominations.won as won
from nominations
inner join ceremonies 
on nominations.year == ceremonies.year
;
'''
joined_nominations = conn.execute(nom_query).fetchall()

conn.execute(create_nominations_two)

insert_nominations_two = '''insert into nominations_two (ceremony_id, category, nominee, movie, character, won) 
values (?,?,?,?,?,?);
'''

conn.executemany(insert_nominations_two, joined_nominations)
print(conn.execute("select * from nominations_two limit 5;").fetchall())

drop_nominations = "drop table nominations;"
conn.execute(drop_nominations)

rename_nominations_two = "alter table nominations_two rename to nominations;"
conn.execute(rename_nominations_two)

create_movies = "create table movies (id integer primary key,movie text);"
create_actors = "create table actors (id integer primary key,actor text);"
create_movies_actors = '''create table movies_actors (id INTEGER PRIMARY KEY,
movie_id INTEGER references movies(id), actor_id INTEGER references actors(id));
'''
conn.execute(create_movies)
conn.execute(create_actors)
conn.execute(create_movies_actors)

insert_movies = "insert into movies (movie) select distinct movie from nominations;"
insert_actors = "insert into actors (actor) select distinct nominee from nominations;"
conn.execute(insert_movies)
conn.execute(insert_actors)

print(conn.execute("select * from movies limit 5;").fetchall())
print(conn.execute("select * from actors limit 5;").fetchall())

pairs_query = "select movie,nominee from nominations;"
movie_actor_pairs = conn.execute(pairs_query).fetchall()

join_table_insert = "insert into movies_actors (movie_id, actor_id) values ((select id from movies where movie == ?),(select id from actors where actor == ?));"
conn.executemany(join_table_insert,movie_actor_pairs)

print(conn.execute("select * from movies_actors limit 5;").fetchall())



