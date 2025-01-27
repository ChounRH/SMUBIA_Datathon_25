import pandas as pd
from nlp import nlp, extract_entities, extract_relationships, extract_contextual_numerical_data
import json

# Input and output files
INPUT_FILE = "input.xlsx"
OUTPUT_FILE = "NEPprocessed_data.json"

def main():
    print("Loading Excel file...")
    df = pd.read_excel(INPUT_FILE)

    output_data = []

    for idx, summary in enumerate(df["Text"]):
        print(f"Processing row {idx + 1}/{len(df)}...")

        # Extract entities and numerical data
        entities = extract_entities(summary)
        numerical_data = extract_contextual_numerical_data(summary)

        # Extract relationships
        doc = nlp(summary)  # Initialize the SpaCy document here
        relationships = extract_relationships(doc, entities)

        # Append results
        output_data.append({
            "Row": idx + 1,
            "Summary": summary,
            "Entities": [{"Text": ent[0], "Label": ent[1]} for ent in entities],
            "Relationships": relationships,
            "Numerical Data": numerical_data
        })

    # Save results to JSON
    print("Saving results to JSON...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Processing complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
