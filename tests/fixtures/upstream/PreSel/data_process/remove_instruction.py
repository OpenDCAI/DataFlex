import json
from tqdm import tqdm
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Remove text-only instructions and keep only image instructions and GPT responses.")
parser.add_argument('--input_path', type=str, default='../data/round1_665k_notext.json', help='Path to the input JSON file')
parser.add_argument('--output_path', type=str, default='../data/round1_665k_notext_img_token.json', help='Path to save the processed JSON file')
args = parser.parse_args()

# Load the JSON files    
with open(args.input_path, 'r') as f:
    data = json.load(f)

# Preprocess the data
def preprocess(data):
    processed_data = []
    for entry in tqdm(data):
        processed_entry = {
            "id": entry["id"],
            "image": entry["image"],
            "unique_idx": entry["unique_idx"],
            "conversations": []
        }
        gpt_responses = []
        for conv in entry["conversations"]:
            if conv["from"] == "human" and "<image>" in conv["value"]:
                processed_entry["conversations"].append({"from": "human", "value": "<image>"})
            elif conv["from"] == "gpt":
                gpt_responses.append(conv["value"])
        combined_gpt_response = " ".join(gpt_responses)
        processed_entry["conversations"].append({"from": "gpt", "value": combined_gpt_response})
        processed_data.append(processed_entry)
    return processed_data

# Process the data
processed_data = preprocess(data)

with open(args.output_path, 'w') as f:
    json.dump(processed_data, f, indent=4)

print(f"Processed data saved to '{args.output_path}'")
