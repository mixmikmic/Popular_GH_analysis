import requests
import newspaper
from newspaper import Article
from xml.etree import ElementTree
from py2neo import Graph

graph = Graph()

INSERT_ARTICLE_QUERY = '''
    MERGE (u:URL {url: {url}})
    SET u.title = {title}
    FOREACH (keyword IN {keywords} | MERGE (k:Keyword {text: keyword}) CREATE UNIQUE (k)<-[:IS_ABOUT]-(u) )
    FOREACH (img IN {images} | MERGE (i:Image {url: img})<-[:WITH_IMAGE]-(u) )
    FOREACH (vid IN {videos} | MERGE (v:Video {url: vid})<-[:WITH_VIDEO]-(u) )
    FOREACH (author IN {authors} | MERGE (a:Author {name: author})<-[:AUTHORED_BY]-(u) )    
'''

INSERT_LIKED_QUERY = '''
    MERGE (u:User {name: {username}})
    MERGE (a:URL {url: {url}})
    CREATE UNIQUE (u)-[:LIKED]->(a)
'''

# insert liked articles
for u in liked_articles:
    insertLikedArticle("lyonwj", u)
    article = newspaper_article(u)
    writeToGraph(article)
    

# insert newspaper articles
for url in newspapers:
    p = newspaper.build(url)
    for article in p.articles:
        parsed_a = newspaper_article(article.url)
        writeToGraph(parsed_a)

# articles from the read later queue
liked_articles = [
    'http://paulgraham.com/ineq.html',
    'https://codewords.recurse.com/issues/five/what-restful-actually-means',
    'http://priceonomics.com/the-history-of-the-black-scholes-formula/',
    'https://buildingrecommenders.wordpress.com/2015/11/16/overview-of-recommender-algorithms-part-1/',
    'http://blog.crew.co/makers-and-managers/',
    'http://www.lrb.co.uk/v37/n22/jacqueline-rose/bantu-in-the-bathroom',
    'http://www.techrepublic.com/article/how-the-paypal-mafia-redefined-success-in-silicon-valley/',
    'http://www.bloomberg.com/bw/articles/2012-07-10/how-the-mormons-make-money',
    'https://jasonrogena.github.io/2015/10/09/matatus-route-planning-using-neo4j.html',
    'http://efavdb.com/principal-component-analysis/',
    'http://www.tsartsaris.gr/How-to-write-faster-from-Python-to-Neo4j-with-OpenMpi',
    'http://burakkanber.com/blog/machine-learning-full-text-search-in-javascript-relevance-scoring/',
    'https://www.pubnub.com/blog/2015-10-22-turning-neo4j-realtime-database/',
    'http://www.greatfallstribune.com/story/news/local/2016/01/12/montana-coal-mine-deal-includes-secret-side-settlement/78697796/',
    'http://billingsgazette.com/news/opinion/editorial/gazette-opinion/a-big-win-for-montana-businesses-taxpayers/article_ffa8c111-ce4b-508f-8813-8337b6d9a4b2.html',
    'http://billingsgazette.com/news/state-and-regional/montana/appeals-court-says-one-time-billionaire-will-stay-in-montana/article_90e41f92-60a5-5685-90ba-ad63721715c7.html',
    'http://missoulian.com/news/state-and-regional/missoula-man-seeks-a-fortune-in-anaconda-slag/article_c1fa2a2a-3468-56fe-a794-814f83a8eb6a.html',
    'http://www.theverge.com/2015/9/30/9416579/spotify-discover-weekly-online-music-curation-interview',
    'https://theintercept.com/2015/09/09/makers-zero-dark-thirty-seduced-cia-tequila-fake-earrings/',
    'https://www.quantamagazine.org/20150903-the-road-less-traveled/',
    'https://medium.com/@bolerio/scheduling-tasks-and-drawing-graphs-the-coffman-graham-algorithm-3c85eb975ab#.xm0lpx2l3',
    'http://www.datastax.com/dev/blog/tales-from-the-tinkerpop',
    'http://open.blogs.nytimes.com/2015/08/11/building-the-next-new-york-times-recommendation-engine/?_r=0',
    'http://www.economist.com/news/americas/21660149-voters-are-about-start-choosing-next-president-scion-and-heir?fsrc=scn/tw/te/pe/ed/TheScionAndTheHeir',
    'https://lareviewofbooks.org/essay/why-your-rent-is-so-high-and-your-pay-is-so-low-tom-streithorst',
    'http://www.economist.com/news/asia/21660551-propaganda-socialist-theme-park-relentless-so-march-money-bread-and-circuses?fsrc=scn/tw/te/pe/ed/BreadAndCircuses',
    'http://www.markhneedham.com/blog/2015/08/10/neo4j-2-2-3-unmanaged-extensions-creating-gzipped-streamed-responses-with-jetty/?utm_source=NoSQL+Weekly+Newsletter&utm_campaign=5836be97da-NoSQL_Weekly_Issue_246_August_13_2015&utm_medium=email&utm_term=0_2f0470315b-5836be97da-328632629',
    'https://medium.com/@dtauerbach/software-engineers-will-be-obsolete-by-2060-2a214fdf9737#.lac4umwmq',
    'http://www.nytimes.com/2015/08/16/opinion/sunday/how-california-is-winning-the-drought.html?action=click&pgtype=Homepage&module=opinion-c-col-right-region&region=opinion-c-col-right-region&WT.nav=opinion-c-col-right-region&_r=1'
]

# source for potential articles to recommend
newspapers = [
    'http://cnn.com',
    'http://news.ycombinator.com',
    'http://nytimes.com',
    'http://missoulian.com',
    'http://www.washingtonpost.com',
    'http://www.reuters.com/',
    'http://sfgate.com',
    'http://datatau.com',
    'http://economist.com',
    'http://medium.com',
    'http://theverge.com'
]



def insertLikedArticle(username, url):
    graph.cypher.execute(INSERT_LIKED_QUERY, {"username": username, "url": url})

def writeToGraph(article):

    #TODO: better data model, remove unnecessary data from data model
    insert_tx = graph.cypher.begin()
    insert_tx.append(INSERT_ARTICLE_QUERY, article)
    insert_tx.commit()



def newspaper_article(url):
    
    article = Article(url)
    article.download()
    article.parse()

    try:
        html_string = ElementTree.tostring(article.clean_top_node)
    except:
        html_string = "Error converting HTML to string"

    try:
        article.nlp()
    except:
        pass

    return {
        'url': url,
        'authors': article.authors,
        'title': article.title,
        'top_image': article.top_image,
        'videos': article.movies,
        'keywords': article.keywords,
        'images': filter_images(list(article.images))
    }



def filter_images(images):
    imgs = []
    for img in images:
        if img.startswith('http'):
            imgs.append(img)
    return imgs

# TODO: generate recommendations





