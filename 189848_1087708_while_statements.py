import random

deaths = 6

running = True

while running:
    # Create a variable that randomly create a integer between 0 and 10.
    guess = random.randint(0,10)

    # if guess equals deaths,
    if guess == deaths:
        # then print this
        print('Correct!')
        # and then also change running to False to stop the script
        running = False
    # else if guess is lower than deaths
    elif guess < deaths:
        # then print this
        print('No, it is higher.')
    # if guess is none of the above
    else:
        # print this
        print('No, it is lower')

