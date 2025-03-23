import json
import csv
from collections import defaultdict
import pandas as pd

def extract_stv(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Function to clean and parse JSON
    def clean_and_parse_json(data_str):
        if pd.isna(data_str):  # Handle NaN values
            return None
        try:
            data_str = data_str.replace("'", '"')
            data_str = data_str.replace("None", 'null')
            data_str = data_str.replace("True", 'true')
            data_str = data_str.replace("False", 'false')
            return json.loads(data_str)
        except json.JSONDecodeError:
            return None  # Return None if JSON parsing fails

    # Apply transformation to all rows in 'Trace ID' column
    df['Trace ID'] = df['Trace ID'].apply(clean_and_parse_json)
    stv= []
    weird_JSON=['5dfbd16922dd6701', '4eec3a2951c10f34', '57562e2a55c097e1', 'a5377dfa9197cd34', '5c6a25aada276463', 'e18a0f08ea47cb72', '4f02e9537e8a95f9', 'bb28bbaef00cf149', '22bc2e26f0ab1dc4', '5107e047bec34fbd', '19e28f7b270ff462', '196031a5c3623549', 'd2e63e1b25b4b661', '160d0bce932aad75', '10b1e4d4c82644d6', 'b9adaaca4a412fcc', '005e7c1e81a9b54f', '12d57eb9cad06bd2', 'cfdb3057e318b426', 'daff9433c31777ee', '55fa6a19b5e221c7', '982923cb5ae1a2a5', '30c7abe403b1e4bb', '3bd0b56a92772176', '50472f2dedb5309b', '09b28f1036f41153', 'af9eb28f5ed8c224', 'b99e4a9c8a950e69', '5ed0f854ab589ffe', '0481611e06f414cc', '5d242e7b7b342484', '90ca15970d08f604', 'f8a9519ae3848c73', 'af3f0164e530961c', 'a95dae24abe19c51', '80287b426a683795']
    
    for row in range(len(df)):
        
        tstv = []
        trace_id = df['Trace ID'][row].get("traceID", "Unknown")
        spans = df['Trace ID'][row].get("spans", [])
        if trace_id in weird_JSON:
            with open("json\\" + trace_id + ".json", "w") as file:
                json.dump(df['Trace ID'][row], file, indent=4)
        
        # Mapping spanID to parent and operation
        span_map = {}
        for span in spans:
            span_id = span["spanID"]
            parent_id = span["references"][0]["spanID"] if span["references"] else None
            operation = span["operationName"]
            start_time = span["startTime"]
            duration = span["duration"]
            response_time = duration  # Assuming response time is duration
            
            span_map[span_id] = {"parent": parent_id, "operation": operation, "response_time": response_time}
        
        # Construct call paths
        call_paths = {}
        for span_id, info in span_map.items():
            call_path = []
            current_id = span_id
            
            while current_id:
                call_path.insert(0, span_map[current_id]["operation"])
                current_id = span_map[current_id]["parent"]
            
            call_paths[span_id] = " → ".join(call_path)
        
        # Create STV
        for span_id, path in call_paths.items():
            tstv.append({"call_path": path, "response_time": span_map[span_id]["response_time"]})

        
        stv.append({
        "trace_id": trace_id,
        "stv": tstv,
        "anomaly_score": df['Anomalous'][row]
        })
        
    return stv

# Example usage
in_csv_file="data\problem-write_home_timeline_server-2500-250.0-20250320_163745-traces.csv"
out_csv_file="data\STVs.csv"
stv = extract_stv(in_csv_file)
file = open(out_csv_file, mode="w", newline="")
writer = csv.writer(file)
writer.writerow(["trace_id","stv","anomaly_score"])
max_len_of_processes=0

for i in stv:
    if len (i["stv"])>max_len_of_processes:
        max_len_of_processes=len (i["stv"])
weird_JSON=[]
for i in stv:
    t_stv_array=[]
    flag=0
    for j in range(max_len_of_processes):
        try:
           t_stv_array.append( i["stv"][j]["response_time"])
        except:
            flag=1
            t_stv_array.append(0)
    if flag:
        weird_JSON.append(i["trace_id"])
    
    writer.writerow([i["trace_id"],t_stv_array,i["anomaly_score"]])
            
            
    
        
file.close()
print(weird_JSON)

