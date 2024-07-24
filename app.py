import pandas as pd
import matplotlib.pyplot as plt

# Manually defined average values for Reranker and Vector Search
reranker_values = [0.87, 0.97, 0.99, 0.89, 0.88, 0.61, 0]  # Example values
vector_search_values = [0.96, 0.97, 0.99, 0.85, 0.89, 0.53, 0]  # Example values

# Define evaluation parameters
eval_parameters = [
    "Faithfulness",
    "Answer Relevancy",
    "Context Precision",
    "Context Recall",
    "Answer Similarity",
    "Answer Correctness",
    "Harmfulness"
]

# Creating a DataFrame with these series and evaluation parameters as index
average_data = pd.DataFrame({
    'Reranker': reranker_values,
    'Vector Search': vector_search_values
}, index=eval_parameters)

# Convert values to percentages for clearer interpretation
average_data_percentage = average_data * 100

# Plotting the data
plt.figure(figsize=(20, 8))  # Adjust the figure size
average_data_percentage.plot(kind='bar')
plt.title('Evaluation Parameter values for Reranker vs. Vector Search')
plt.xlabel('Evaluation Parameters')
plt.ylabel('Values(%)')
plt.legend()
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.subplots_adjust(bottom=0.4)  # Adjust the bottom margin to prevent cutting off text
plt.show()
