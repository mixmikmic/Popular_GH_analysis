pipeline = [lambda x: x **2 - 1 + 5,
            lambda x: x **20 - 2 + 3,
            lambda x: x **200 - 1 + 4]

for f in pipeline:
    print(f(3))

