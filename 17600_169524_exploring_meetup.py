import json
import mimeparse
import requests
import urllib
import pandas as pd
from pprint import pprint as pp

MEETUP_API_HOST = 'https://api.meetup.com'
EVENTS_URL = MEETUP_API_HOST + '/2/events.json'
MEMBERS_URL = MEETUP_API_HOST + '/2/members.json'
GROUPS_URL = MEETUP_API_HOST + '/2/groups.json'
RSVPS_URL = MEETUP_API_HOST + '/2/rsvps.json'
PHOTOS_URL = MEETUP_API_HOST + '/2/photos.json'
GROUP_URLNAME = 'London-Machine-Learning-Meetup'

# GROUP_URLNAME = 'Data-Science-London'

# Load Meetup API Key
meetup_api_key = pd.read_csv("../meetup_token.csv")

class MeetupAPI(object):
    """ Retreives information about meetup.com
    """
    def __init__(self, api_key, num_past_events=10, http_timeout=1, http_retries=2):
        """ Create new instance of meetup """
        self._api_key = api_key
        self._http_timeout = http_timeout
        self._http_retries = http_retries
        self._num_past_events = num_past_events

    
    def get_past_events(self):
        """ Get past meetup events for a given meetup group """
        params = {'key': self._api_key,
                  'group_urlname': GROUP_URLNAME,
                  'status': 'past',
                  'desc': 'true'}
        if self._num_past_events:
            params['page'] = str(self._num_past_events)
            
        query = urllib.urlencode(params)
        url = '{0}?{1}'.format(EVENTS_URL, query)
        response = requests.get(url, timeout=self._http_timeout)
        data = response.json()['results']
        return data
    
    

    def get_members(self):
        """ Get meetup members for a given meetup group """
        params = {'key': self._api_key,
                  'group_urlname': GROUP_URLNAME,
                  'offset': '0',
                  'format': 'json',
                  'page': '100',
                  'order':'name'}
        query = urllib.urlencode(params)
        url = '{0}?{1}'.format(MEMBERS_URL, query)
        response = requests.get(url, timeout=self._http_timeout)
        data = response.json()['results']
        return data
    
    
    def get_groups_by_member(self, member_id='38680722'):
        """Get meetup groups for a given meetup member """
        params = {'key': self._api_key,
                  'member_id': member_id,
                  'offset': '0',
                  'format':'json',
                  'page':'100',
                  'order':'id'}
        query = urllib.urlencode(params)
        url = '{0}?{1}'.format(GROUPS_URL, query)
        response = requests.get(url, timeout=self._http_timeout)
        data = response.json()['results']
        return data
    
    

m = MeetupAPI(api_key=meetup_api_key.values.flatten()[0])

last_meetups = m.get_past_events()
pp(last_meetups[1])

members = m.get_members()
members



