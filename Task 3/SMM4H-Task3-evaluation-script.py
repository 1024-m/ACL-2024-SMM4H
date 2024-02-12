import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def load_data(file_path):
    """
    Load the prediction or reference data from a TSV file.
    """
    return pd.read_csv(file_path, sep='\t')

def evaluate(predictions, references):
    """
    Evaluate the micro-averaged precision, recall, and F1 score.
    """
    # Merge predictions and references on the 'id' column
    merged_data = pd.merge(predictions, references, on='id')

    # Calculate micro-averaged precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(merged_data['Label_y'], merged_data['Label_x'], average='macro')

    # Calculate accuracy for reference
    accuracy = accuracy_score(merged_data['Label_y'], merged_data['Label_x'])

    return precision, recall, f1, accuracy

if __name__ == "__main__":
    # Specify file paths for prediction and reference data
    prediction_file = "/path/to/predictions.tsv"
    reference_file = "/path/to/reference_data_0/class.tsv"

    # Load prediction and reference data
    predictions = load_data(prediction_file)
    references = load_data(reference_file)

    # Evaluate and get micro-averaged precision, recall, and F1 score
    precision, recall, f1, accuracy = evaluate(predictions, references)

    # Print the results
    print(f"Micro-Averaged Precision: {precision}")
    print(f"Micro-Averaged Recall: {recall}")
    print(f"Micro-Averaged F1 Score: {f1}")
    print(f"Accuracy: {accuracy}")

