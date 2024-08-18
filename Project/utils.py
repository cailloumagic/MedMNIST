

def get_memory_and_duration(start_time):
    # Returns the memory usage and duration since the provided start time.
    end_time = time.time()
    training_duration = end_time - start_time   # Calculate the duration
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info, training_duration