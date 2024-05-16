from distributed import Client
from time import sleep
import random
import math

cl = Client()

cl

def nsleep(num_secs):
    sleep(num_secs)
    return num_secs

def get_rand_secs(max_int):
    return random.randint(0, max_int)

def do_some_math_with_errors(number):
    if number > 30:
        return math.log(number)
    elif number > 15:
        return round(number / 3.0)
    elif number >= 5:
        return math.floor(number / 1.5)
    elif number <= 2:
        return number / 0
    return number ** 2

def do_some_math(number):
    if number > 30:
        return math.log(number)
    elif number > 15:
        return round(number / 3.0)
    elif number >= 5:
        return math.floor(number / 1.5)
    elif number <= 2:
        return number 
    return number ** 2

random_secs = cl.map(get_rand_secs, range(200))

random_secs

cl.gather(random_secs)[:10]

random_math = cl.map(do_some_math_with_errors, random_secs)

excs = [(e.traceback(), e.exception()) for e in random_math if e.exception()]

excs[0]

random_math = cl.map(do_some_math, random_secs)

random_sleeps = cl.map(nsleep, random_math)

random_sleeps

sum_sleep = cl.submit(sum, random_sleeps)

sum_sleep

sum_sleep.result()



