import chatterbot
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer

#Let's give a name to your chat bot
chatbot = ChatBot('Veronica')

#Train the bot
chatbot.set_trainer(ListTrainer)
chatbot.train(['What is your name?', 'My name is Veronica'])

#let's test the bot
chatbot.get_response('What is your name?')

#Let's train the bot with more data
conversations = [
     'Are you an artist?', 'No, are you mad? I am a bot',
     'Do you like big bang theory?', 'Bazinga!',
     'What is my name?', 'Natasha',
     'What color is the sky?', 'Blue, stop asking me stupid questions'
]

chatbot.train(conversations)

chatbot.get_response('Do you like big bang theory?')

#Let's use chatterbot corpus to train the bot
from chatterbot.trainers import ChatterBotCorpusTrainer
chatbot.set_trainer(ChatterBotCorpusTrainer)
chatbot.train("chatterbot.corpus.english")

chatbot.get_response('Who is the President of America?')

chatbot.get_response('What language do you speak?')

chatbot.get_response('Hi')

