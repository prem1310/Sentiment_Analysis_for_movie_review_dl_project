import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, Frame, Label, Button, Entry, Text, filedialog, Scrollbar, RIGHT, Y, END

classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", truncation=True)

def analyze_single_review():
    review = user_input.get()
    result = classifier(review)[0]
    label = result['label']
    if label == '1 star' or label == '2 stars':
        sentiment = 'Negative'
    elif label == '3 stars':
        sentiment = 'Neutral'
    elif label == '4 stars' or label == '5 stars':
        sentiment = 'Positive'
    result_text.delete("1.0", END)
    result_text.insert(END, f"Review: {review}\nSentiment: {sentiment} (Score: {result['score']:.2f})\n")

def analyze_csv_file():
    batch_size = 10
    file_path = filedialog.askopenfilename()
    if file_path:
        df = pd.read_csv(file_path)
        reviews = df['text'].tolist()
        results_list = []
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i+batch_size]
            results = classifier(batch)
            for review, result in zip(batch, results):
                label = result['label']
                if label == '1 star' or label == '2 stars':
                    sentiment = 'Negative'
                elif label == '3 stars':
                    sentiment = 'Neutral'
                elif label == '4 stars' or label == '5 stars':
                    sentiment = 'Positive'
                results_list.append({
                    "Review": review,
                    "Sentiment": sentiment,
                    "Score": result['score']
                })
        results_df = pd.DataFrame(results_list)
        sentiment_counts = results_df['Sentiment'].value_counts()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
        plt.title('Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.show()

        result_text.delete("1.0", END)
        for index, row in results_df.iterrows():
            result_text.insert(END, f"Review: {row['Review']}\nSentiment: {row['Sentiment']} (Score: {row['Score']:.2f})\n-----\n")

root = Tk()
root.title("Sentiment Analysis")
root.geometry("800x600")

l1 = Label(root, text="Enter your review:")
l1.pack(padx=10, pady=5)
user_input = Entry(root, width=50)
user_input.pack(padx=10, pady=5)
b1 = Button(root, text="Analyze Review", command=analyze_single_review)
b1.pack(padx=10, pady=5)

l2 = Label(root, text="Or upload a CSV file:")
l2.pack(padx=10, pady=5)
b2 = Button(root, text="Upload CSV", command=analyze_csv_file)
b2.pack(padx=10, pady=5)

frame = Frame(root)
frame.pack(padx=10, pady=5, expand=True, fill='both')

result_text = Text(frame, height=20, width=80)
scrollbar = Scrollbar(frame, command=result_text.yview)
result_text.config(yscrollcommand=scrollbar.set)

result_text.pack(side='left', fill='both', expand=True)
scrollbar.pack(side=RIGHT, fill=Y)

root.mainloop()