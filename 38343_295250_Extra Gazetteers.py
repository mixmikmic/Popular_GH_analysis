get_ipython().run_cell_magic('bash', '', 'cd data/extra_gazetteers/\nwget ftp://ftp.fu-berlin.de/pub/misc/movies/database/movies.list.gz\nzcat movies.list.gz | cut -f1 -d\'(\' | sed -s \'s/\\"//g\' | sort | uniq > all_movie_titles.txt\necho $(wc -l all_movie_titles.txt)\ncp all_movie_titles.txt ../cleaned/custom_lexicons/')

get_ipython().system(' wc -l data/cleaned/custom_lexicons/all_movie_titles.txt')

get_ipython().run_cell_magic('bash', '', 'cd data/extra_gazetteers/\nwget http://discogs-data.s3-us-west-2.amazonaws.com/data/discogs_20160901_artists.xml.gz')

from bs4 import BeautifulSoup
import gzip
import codecs

with gzip.open("data/extra_gazetteers/discogs_20160901_artists.xml.gz") as fp,     codecs.open("data/extra_gazetteers/musicartist_names.txt", "wb+", "utf-8") as fp1,     codecs.open("data/extra_gazetteers/musicartist_namevariants.txt", "wb+", "utf-8") as fp2:
    i = 0
    for line in fp:
        if i == 0:
            i += 1
            continue
        line = line.strip()
        i += 1
        try:
            p = BeautifulSoup(line, "lxml-xml")
            artist_names = p.select("artist > name, aliases > name, groups > name")
            artist_name_variants = p.select("namevariations > name")
            for k in artist_names:
                print >> fp1, unicode(k.text)
            for k in artist_name_variants:
                print >> fp2, unicode(k.text)
        except:
            print "i=%s" % i
    print "i=%s" % i

get_ipython().system(' sort data/extra_gazetteers/musicartist_names.txt | uniq > data/extra_gazetteers/musicartist_names.unique.txt')
get_ipython().system(' sort data/extra_gazetteers/musicartist_namevariants.txt | uniq > data/extra_gazetteers/musicartist_namevariants.unique.txt')

get_ipython().system(' cp data/extra_gazetteers/*.unique.txt data/cleaned/custom_lexicons/')



