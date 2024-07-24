import panda as pd
from rouge import Rouge
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


# Function to load the data
def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

# Function to calculate BLEU, ROUGE, and METEOR scores
def calculate_scores(reference, candidate):
    # BLEU score
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference.split()], candidate.split(), smoothing_function=smoothing)
    
    # METEOR score
    meteor = meteor_score([reference], candidate)
    
    # ROUGE score
    rouge = Rouge()
    rouge_scores = rouge.get_scores(candidate, reference, avg=True)
    
    return bleu_score, meteor, rouge_scores

# Example usage
def main():
    file_path = 'L2.xlsx'
    sheet_name = 'L2'
    
    data = load_data(file_path, sheet_name)
    data = data.dropna(subset=['Question', 'Answer', 'Flat_search'])  # Adjust columns as per your data structure
    
    results = data.apply(lambda row: calculate_scores(row['Answer'], row['Flat_search']), axis=1)
    
    bleu_scores, meteor_scores, rouge_scores = zip(*results)
    
    average_bleu = sum(bleu_scores) / len(bleu_scores)
    average_meteor = sum(meteor_scores) / len(meteor_scores)
    
    # Averaging ROUGE scores
    average_rouge = {
        'rouge-1': sum([score['rouge-1']['f'] for score in rouge_scores]) / len(rouge_scores),
        'rouge-2': sum([score['rouge-2']['f'] for score in rouge_scores]) / len(rouge_scores),
        'rouge-l': sum([score['rouge-l']['f'] for score in rouge_scores]) / len(rouge_scores)
    }
    
    print(f"Average BLEU Score: {average_bleu}")
    print(f"Average METEOR Score: {average_meteor}")
    print(f"Average ROUGE Scores: {average_rouge}")

if __name__ == "__main__":
    main()
