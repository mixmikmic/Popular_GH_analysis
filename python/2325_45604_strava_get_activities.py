strava_oauth_code = "7ba92a5340010f4035b2f897a7c93d6a9a331b53"

import requests

payload = {
    'client_id':"5966",
    'client_secret':"b8869c83423df058bbd72319cef18bd46123b251",
    'code':strava_oauth_code
}
resp = requests.post("https://www.strava.com/oauth/token", params=payload)
assert resp.status_code == 200

access_token = resp.json()['access_token']
headers = {
    'Authorization': "Bearer " + access_token
}

access_token

resp = requests.get("https://www.strava.com/api/v3/athlete", headers=headers)
assert resp.status_code == 200
athlete = resp.json()
print athlete['firstname'], athlete['lastname']

def get_activities(page):
    params = {
        'per_page': 50,
        'page':page
    }

    resp = requests.get("https://www.strava.com/api/v3/athlete/activities",
                        params=params, headers=headers)
    assert resp.status_code == 200
    activities = resp.json()
    return activities

def get_all_activities():
    all_activities = []
    page = 1
    while True:
        activities = get_activities(page)
        page += 1
        if len(activities) == 0:
            break
        all_activities += activities
    return all_activities
    
activities = get_all_activities()
print len(activities), ' activities total'

import json
with open('activities.json', 'w') as f:
    json.dump(activities, f)

