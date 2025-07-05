
import time, socket

HOSTNAME = socket.gethostname()
def report_duration(start_time, model_name, method, args=''):
    duration = time.time() - start_time
    print(f"DURATION,{HOSTNAME},{model_name},{method},{args},{duration}")
