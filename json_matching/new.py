# %% Imports
import os
import json
from deepdiff import DeepDiff
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein

# Paths to the directories
folder1 = 'D:/Dataset Creation/json_matching/gpt4output'
folder2 = 'D:/Dataset Creation/json_matching/ground_truth_radar chart'

# List all JSON files in both directories
files1 = [f for f in os.listdir(folder1) if f.endswith('.json')]
files2 = [f for f in os.listdir(folder2) if f.endswith('.json')]

# Ensure both directories have the same JSON file names for comparison
common_files = set(files1).intersection(files2)

# %% Function Definitions

# Function to load JSON files
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {file_path}")
        return None

# DeepDiff comparison
def compare_json_files(json1, json2):
    diff = DeepDiff(json1, json2, ignore_order=True, report_repetition=True, view='tree')
    return diff

# Exact match check (ignores list order)
def exact_match(json1, json2):
    return json1 == json2

# Key match comparison
def key_match(json1, json2):
    if isinstance(json1, dict) and isinstance(json2, dict):
        keys1 = set(json1.keys())
        keys2 = set(json2.keys())
        common_keys = keys1.intersection(keys2)
        matching_values = sum(1 for key in common_keys if json1[key] == json2[key])
        return (matching_values / len(common_keys)) * 100 if common_keys else 0.0
    elif isinstance(json1, list) and isinstance(json2, list):
        # Handle list of dicts cases
        if json1 and isinstance(json1[0], dict) and json2 and isinstance(json2[0], dict):
            matching_dicts = [key_match(item1, item2) for item1 in json1 for item2 in json2 if isinstance(item1, dict) and isinstance(item2, dict)]
            return sum(matching_dicts) / len(matching_dicts) if matching_dicts else 0.0
    return 0.0

# Value match comparison
def value_match(json1, json2):
    if isinstance(json1, list) and isinstance(json2, list):
        min_length = min(len(json1), len(json2))
        matching_values = sum(1 for i in range(min_length) if json1[i] == json2[i])
        return matching_values / min_length * 100 if min_length > 0 else 0.0
    elif isinstance(json1, dict) and isinstance(json2, dict):
        common_keys = set(json1.keys()).intersection(json2.keys())
        matching_values = sum(1 for key in common_keys if json1[key] == json2[key])
        return matching_values / len(common_keys) * 100 if common_keys else 0.0
    elif isinstance(json1, list) and isinstance(json2, list):
        return jaccard_similarity(json1, json2)  # Use Jaccard similarity for lists
    return 0.0

# Structural similarity comparison
def structural_similarity(json1, json2):
    diff = DeepDiff(json1, json2, ignore_order=True)
    total_elements = len(json1) + len(json2)
    mismatched_elements = len(diff)
    if total_elements == 0:
        return 100
    return ((total_elements - mismatched_elements) / total_elements) * 100

# Cosine similarity (for text fields)
def cosine_similarity_values(str1, str2):
    count_vectorizer = CountVectorizer().fit_transform([str(str1), str(str2)])
    vectors = count_vectorizer.toarray()
    return cosine_similarity(vectors)[0][1] * 100

# Numerical difference comparison
def numerical_difference(json1, json2):
    if isinstance(json1, dict) and isinstance(json2, dict):
        common_keys = set(json1.keys()).intersection(json2.keys())
        total_difference = 0.0
        count = 0
        for key in common_keys:
            if isinstance(json1[key], (int, float)) and isinstance(json2[key], (int, float)):
                total_difference += abs(json1[key] - json2[key])
                count += 1
        return total_difference / count if count > 0 else 0.0
    return 0.0

# Levenshtein similarity for text fields
def levenshtein_similarity(str1, str2):
    distance = Levenshtein.distance(str(str1), str(str2))
    max_len = max(len(str1), len(str2))
    return (1 - distance / max_len) * 100 if max_len > 0 else 100
# Jaccard similarity for lists
def jaccard_similarity(list1, list2):
    if isinstance(list1, list) and isinstance(list2, list):
        # Handle lists
        set1 = set(map(str, list1))  # Convert elements to strings to handle non-hashable types
        set2 = set(map(str, list2))
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return (intersection / union) * 100 if union > 0 else 100
    elif isinstance(list1, dict) and isinstance(list2, dict):
        # Handle dictionaries by comparing their keys
        set1 = set(list1.keys())
        set2 = set(list2.keys())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return (intersection / union) * 100 if union > 0 else 100
    return 0.0

# Main comparison loop
for file_name in common_files:
    file1_path = os.path.join(folder1, file_name)
    file2_path = os.path.join(folder2, file_name)
    
    print(f"Comparing {file1_path} and {file2_path}")

    json1 = load_json(file1_path)
    json2 = load_json(file2_path)

    if json1 is None or json2 is None:
        continue  # Skip if either JSON is invalid

    with open('D:/Dataset Creation/json_matching/result.txt', 'a') as f:
        f.write(f"Comparing {file1_path} and {file2_path}\n")

        # DeepDiff
        diff_result = compare_json_files(json1, json2)
        f.write("DeepDiff:\n")
        f.write(str(diff_result) + '\n\n')

        # Exact match
        exact_result = exact_match(json1, json2)
        f.write("Exact Match:\n")
        f.write(str(exact_result) + '\n\n')

        # Key match
        key_result = key_match(json1, json2)
        f.write("Key Match Percentage:\n")
        f.write(str(key_result) + '\n\n')

        # Value match
        value_result = value_match(json1, json2)
        f.write("Value Match Percentage:\n")
        f.write(str(value_result) + '\n\n')

        # Structural similarity
        structural_result = structural_similarity(json1, json2)
        f.write("Structural Similarity Percentage:\n")
        f.write(str(structural_result) + '\n\n')

        # Cosine similarity
        cosine_result = cosine_similarity_values(json.dumps(json1), json.dumps(json2))
        f.write("Cosine Similarity Percentage:\n")
        f.write(str(cosine_result) + '\n\n')

        # Numerical difference
        numerical_result = numerical_difference(json1, json2)
        f.write("Numerical Difference:\n")
        f.write(str(numerical_result) + '\n\n')

        # Levenshtein similarity
        levenshtein_result = levenshtein_similarity(json.dumps(json1), json.dumps(json2))
        f.write("Levenshtein Similarity Percentage:\n")
        f.write(str(levenshtein_result) + '\n\n')

        # Jaccard similarity
        jaccard_result = jaccard_similarity(json1, json2)
        f.write("Jaccard Similarity Percentage:\n")
        f.write(str(jaccard_result) + '\n\n')

        f.write("__________________________________________________________\n\n")
