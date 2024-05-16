import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn.apionly as sns

from datetime import datetime

get_ipython().magic('matplotlib inline')
plt.style.use('bmh')
colors = ['#348ABD', '#A60628', '#7A68A6', '#467821', '#D55E00', 
          '#CC79A7', '#56B4E9', '#009E73', '#F0E442', '#0072B2']

# Import json data
with open('data/Hangouts.json') as json_file:
    json_data = json.load(json_file)

# Generate map from gaia_id to real name
def user_name_mapping(data):
    user_map = {'gaia_id': ''}
    for state in data['conversation_state']:
        participants = state['conversation_state']['conversation']['participant_data']
        for participant in participants:
            if 'fallback_name' in participant:
                user_map[participant['id']['gaia_id']] = participant['fallback_name']

    return user_map

user_dict = user_name_mapping(json_data)

# Parse data into flat list
def fetch_messages(data):
    messages = []
    for state in data['conversation_state']:
        conversation_state = state['conversation_state']
        conversation = conversation_state['conversation']
        conversation_id = conversation_state['conversation']['id']['id']
        participants = conversation['participant_data']

        all_participants = []
        for participant in participants:
            if 'fallback_name' in participant:
                user = participant['fallback_name']
            else:
                # Scope to call G+ API to get name
                user = participant['id']['gaia_id']
            all_participants.append(user)
            num_participants = len(all_participants)
        
        for event in conversation_state['event']:
            try:
                sender = user_dict[event['sender_id']['gaia_id']]
            except:
                sender = event['sender_id']['gaia_id']
            
            timestamp = datetime.fromtimestamp(float(float(event['timestamp'])/10**6.))
            event_id = event['event_id']

            if 'chat_message' in event:
                content = event['chat_message']['message_content']
                if 'segment' in content:
                    segments = content['segment']
                    for segment in segments:
                        if 'text' in segment:
                            message = segment['text']
                            message_length = len(message)
                            message_type = segment['type']
                            if len(message) > 0:
                                messages.append((conversation_id,
                                                 event_id, 
                                                 timestamp, 
                                                 sender, 
                                                 message,
                                                 message_length,
                                                 all_participants,
                                                 ', '.join(all_participants),
                                                 num_participants,
                                                 message_type))

    messages.sort(key=lambda x: x[0])
    return messages

# Parse data into data frame
cols = ['conversation_id', 'event_id', 'timestamp', 'sender', 
        'message', 'message_length', 'participants', 'participants_str', 
        'num_participants', 'message_type']

messages = pd.DataFrame(fetch_messages(json_data), columns=cols).sort(['conversation_id', 'timestamp'])

# Engineer features
messages['prev_timestamp'] = messages.groupby(['conversation_id'])['timestamp'].shift(1)
messages['prev_sender'] = messages.groupby(['conversation_id'])['sender'].shift(1)

# Exclude messages are are replies to oneself (not first reply)
messages = messages[messages['sender'] != messages['prev_sender']]

# Time delay
messages['time_delay_seconds'] = (messages['timestamp'] - messages['prev_timestamp']).astype('timedelta64[s]')
messages = messages[messages['time_delay_seconds'].notnull()]
messages['time_delay_mins'] = np.ceil(messages['time_delay_seconds'].astype(int)/60.0)

# Time attributes
messages['day_of_week'] = messages['timestamp'].apply(lambda x: x.dayofweek)
messages['year_month'] = messages['timestamp'].apply(lambda x: x.strftime("%Y-%m"))
messages['is_weekend'] = messages['day_of_week'].isin([5,6]).apply(lambda x: 1 if x == True else 0)

# Limit to messages sent by me and exclude all messages between me and Alison
messages = messages[(messages['sender'] == 'Mark Regan') & (messages['participants_str'] != 'Alison Darcy, Mark Regan')]

# Remove messages not responded within 60 seconds
# This introduces an issue by right censoring the data (might return to address)
messages = messages[messages['time_delay_seconds'] < 60]

messages.head(1)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(211)

order = np.sort(messages['year_month'].unique())
sns.boxplot(x=messages['year_month'], y=messages['time_delay_seconds'], order=order, orient="v", color=colors[5], linewidth=1, ax=ax)
_ = ax.set_title('Response time distribution by month')
_ = ax.set_xlabel('Month-Year')
_ = ax.set_ylabel('Response time')
_ = plt.xticks(rotation=30)

ax = fig.add_subplot(212)
plt.hist(messages['time_delay_seconds'].values, range=[0, 60], bins=60, histtype='stepfilled', color=colors[0])
_ = ax.set_title('Response time distribution')
_ = ax.set_xlabel('Response time (seconds)')
_ = ax.set_ylabel('Number of messages')

plt.tight_layout()

# excluded some colums from csv output
messages.drop(['participants', 'message', 'participants_str'], axis=1, inplace=True)

# Save csv to data folder
messages.to_csv('data/hangout_chat_data.csv', index=False)

# Apply pretty styles
from IPython.core.display import HTML

def css_styling():
    styles = open("styles/custom.css", "r").read()
    return HTML(styles)
css_styling()



