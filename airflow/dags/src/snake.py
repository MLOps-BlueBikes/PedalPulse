import os 

os.environ['GOOGLE_APPLICATION_CREDENTIALS']="airflow/keys/gcp_key.json/pedalpulse-440019-919eead68e28.json"

print(os.getcwd())

with open(os.environ['GOOGLE_APPLICATION_CREDENTIALS'],'r') as f:
    print(f.read())
    
print(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])