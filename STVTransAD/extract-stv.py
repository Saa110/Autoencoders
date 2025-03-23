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
    
    for row in range(len(df)):
        
        tstv = []
        trace_id = df['Trace ID'][row].get("traceID", "Unknown")
        spans = df['Trace ID'][row].get("spans", [])
        
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
with open(out_csv_file, "w", encoding="utf-8", newline="") as csvfile:
        fieldnames = ["trace_id", "stv", "anomaly_score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stv)

print("Done")
