import os
import pickle
import queue, logging
import threading
from concurrent.futures import ThreadPoolExecutor

# Function for worker threads to process
def pkl_worker(input_queue, output_queue):
    while True:
        try:
            # Get the next path from the queue
            path = input_queue.get(timeout=1)
            # Load the pickled object
            logging.debug(f"loading scors: {path}")
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            # Push the object into the output queue
            output_queue.put(obj)
        except queue.Empty:
            logging.info("Done reading")
            # No more items to process, exit the loop
            break
        except Exception as e:
            logging.exception("error with path {path} : {e}")
            

# Main function
def read_pickle_files(paths, num_workers=20):
    # Queues for input paths and output objects
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    # Push all paths into the input queue
    for path in paths:
        input_queue.put(path)

    # Create a thread pool
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for _ in range(num_workers):
            executor.submit(pkl_worker, input_queue, output_queue)

    # Collect the results from the output queue
    results = []
    while not output_queue.empty() or not input_queue.empty():
        try:
            result = output_queue.get(timeout=1)
            results.append(result)
        except queue.Empty:
            continue

    return results
