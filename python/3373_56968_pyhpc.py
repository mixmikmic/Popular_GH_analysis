import threading
import time

# Generic worker thread
def worker(num):
        
    # Get this Thread's name
    name = threading.currentThread().getName()
    
    # Print Starting Message
    print('{0:s} starting.\n'.format(name), flush=True)
    
    # We sleep for two seconds
    time.sleep(2)
    
    # Print computation
    print('Computation = {0:d}\n'.format(10**num), flush=True)
    
    # Print Exiting Message
    print('{0:s} exiting.\n'.format(name), flush=True)

# We will spawn several threads.
for i in range(5):
    t = threading.Thread(name='Thread #{0:d}'.format(i), target=worker, args=(i,))
    t.start()
    
print("Threads all created", flush=True)

import multiprocessing 
import time

# Generic worker process
def worker(num):
        
    # Get this Process' name
    name = multiprocessing.current_process().name
    
    # Print Starting Message
    print('{0:s} starting.\n'.format(name))
    
    # We sleep for two seconds
    time.sleep(2)
    
    # Print computation
    print('Computation = {0:d}\n'.format(num**10))
    
    # Print Exiting Message
    print('{0:s} exiting.\n'.format(name))

if __name__ == '__main__':

    # We will spawn several processes.
    for i in range(3):
        p = multiprocessing.Process(name='Process #{0:d}'.format(i), target=worker, args=(i,))
        p.start()
        
    print("Processing complete", flush=True)

