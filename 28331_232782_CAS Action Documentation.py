# Load the swat package and turn off note messages
import swat
swat.options.cas.print_messages = False

# set the connection: host, port, username, password
s = swat.CAS(host, port, username, password)

# list all loaded actionsets
s.builtins.actionSetInfo()

# list each actionset with available actions as an ordered dict
s.help()

# session.actionset.action
help(s.dataPreprocess.impute)

# list all of the actionsets, whether they are loaded or not
s.builtins.actionSetInfo(all = True)

# load in new actionset
s.builtins.loadActionSet('decisionTree')

# get help again
s.help().decisionTree

help(s.decisionTree.gbtreeTrain)

s.session.endsession() # end the session

