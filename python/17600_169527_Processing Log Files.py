import datetime
import os, re, sys
import pandas as pd
from pyspark.sql import Row
# from test_helper import Test

# Use Pandas to read sample and inspect
logDataFile = "../_datasets_downloads/NASA_access_log_Aug95.gz"
logfile = pd.read_table(logDataFile, header=None, encoding='utf-8')
# test_log = logfile[0][0]
# test_log
logfile

APACHE_ACCESS_LOG_PATTERN = '^(\S+) (\S+) (\S+) \[([\w:/]+\s[+\-]\d{4})\] "(\S+) (\S+)\s*(\S*)" (\d{3}) (\S+)'



# Search pattern and extract
match = re.search(APACHE_ACCESS_LOG_PATTERN, test_log)
print(match.group(9))

match.group(6)

#  Convert sample `logfile` entry to date time format

month_map = {'Jan': 1, 'Feb': 2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6, 'Jul':7,
    'Aug':8,  'Sep': 9, 'Oct':10, 'Nov': 11, 'Dec': 12}


def parse_apache_time(s):
    """ Convert date entry in log file to datetime format"""
    return datetime.datetime(int(s[7:11]),
                             month_map[s[3:6]],
                             int(s[0:2]),
                             int(s[12:14]),
                             int(s[15:17]),
                             int(s[18:20]))


# Test
parse_apache_time(match.groups()[3])


def parseApacheLogLine(logline):
    """ Parse each line entry of the log file """
    match = re.search(APACHE_ACCESS_LOG_PATTERN, logline)
    
    # If no match is found, return entry and zero
    if match is None:
        return(logline, 0)
    
    # If field_size is empty, initialize to zero
    field_size = match.group(9) 
    if field_size == '_':
        size = long(0)
    else:
        size = long(match.group(9))
    return (Row(
            host = match.group(1),
            client_identd = match.group(2),
            user_id = match.group(3),
            date_time = parse_apache_time(match.group(4)),
            method = match.group(5),
            endpoint = match.group(6),
            protocol = match.group(7),
            response_code = int(match.group(8)),
            content_size = size
            ), 1)

parseApacheLogLine(test_log)

# We first load the text file using sc.textFile(filename) 
# to convert each line of the file into an element in an RDD.

log_fileRDD = sc.textFile("../_datasets_downloads/NASA_access_log_Jul95")
log_fileRDD.take(2)

# Next, we use map(parseApacheLogLine) to apply the parse function to each element 
# (that is, a line from the log file) in the RDD and turn each line into a pair Row object.
log_fileRDD = log_fileRDD.map(parseApacheLogLine)
log_fileRDD.cache()

def parseLogs():
    """ Read and parse log file. """
    parsed_logs = (sc.textFile(logDataFile).map(parseApacheLogLine).cache())
    
    access_logs = (parsed_logs.filter(lambda s: s[1] == 1).map(lambda s: s[0]).cache())
    
    failed_logs = (parsed_logs.filter(lambda s: s[1] == 0).map(lambda s: s[0]))
    
    failed_logs_count = failed_logs.count()
    
    if failed_logs_count > 0:
        print("Number of invalud logline" % failed_logs_count())
        for line in failed_logs.take(20):
            print("Invalid login: %s" %line)
    
    print("Lines red : %d, \n Parsed successfully: %d \n Failed parse: %d"% (parsed_logs.count(),
                                                                            access_logs.count(),
                                                                            failed_logs.count()))
    return parsed_logs, access_logs, failed_logs



parsed_logs, access_logs, failed_logs = parseLogs()



