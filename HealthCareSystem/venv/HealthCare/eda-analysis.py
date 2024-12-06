import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns

# Load intents data
with open("intents.json") as file:
    intents = json.load(file)

# Convert intents to a DataFrame
data = []
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        data.append({'tag': tag, 'pattern': pattern})
df = pd.DataFrame(data)
# Sample Data
print(df.head())

# 1. Overview
print("Total number of intents:", len(intents['intents']))
print("Total number of patterns:", df.shape[0])

# 2. Intent Distribution
intent_counts = df['tag'].value_counts()
print("\nPatterns per Intent:")
print(intent_counts)

# Creating labels and sizes for the pie chart
labels = intent_counts.index  # Access index as labels
sizes = intent_counts.values  # Access values directly

# Plotting the pie chart
plt.figure(figsize=(10, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 7})
plt.title("Distribution of Patterns per Intent")
plt.tight_layout()
plt.savefig("static/eda_plots/intent_distribution_pie_chart.png")
plt.show()

# Plot Intent Distribution
plt.figure(figsize=(12, 6))
sns.barplot(x=intent_counts.index, y=intent_counts.values)
plt.title('Patterns per Intent')
plt.xlabel('Intent')
plt.ylabel('Number of Patterns')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("static/eda_plots/intent_distribution.png")
plt.close()

# 3. Text Analysis
# Pattern length analysis
df['pattern_length'] = df['pattern'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(df['pattern_length'], kde=True, bins=20)
plt.title('Distribution of Pattern Lengths')
plt.xlabel('Pattern Length (Characters)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("static/eda_plots/pattern_length_distribution.png")
plt.close()

# Word frequency
all_words = ' '.join(df['pattern']).split()
word_counts = Counter(all_words)
common_words = word_counts.most_common(20)

# Plot Word Frequency
words, counts = zip(*common_words)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(words), y=list(counts))
plt.title('Top 20 Most Common Words')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("static/eda_plots/word_frequency.png")
plt.close()

# Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['pattern']))
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Patterns')
plt.tight_layout()
plt.savefig("static/eda_plots/word_cloud.png")
plt.close()

# 4. Additional Analyses
# Cumulative distribution of intent patterns
plt.figure(figsize=(12, 6))
sns.ecdfplot(data=intent_counts.values, label="Cumulative Distribution")
plt.title('Cumulative Distribution of Patterns per Intent')
plt.xlabel('Number of Patterns')
plt.ylabel('Cumulative Frequency')
plt.legend()
plt.tight_layout()
plt.savefig("static/eda_plots/cumulative_distribution.png")
plt.close()

# Average pattern length per intent
avg_pattern_length = df.groupby('tag')['pattern_length'].mean().sort_values()

plt.figure(figsize=(12, 6))
sns.barplot(x=avg_pattern_length.index, y=avg_pattern_length.values)
plt.title('Average Pattern Length per Intent')
plt.xlabel('Intent')
plt.ylabel('Average Pattern Length')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("static/eda_plots/average_pattern_length.png")
plt.close()

# Character frequency
all_characters = ''.join(df['pattern'])
character_counts = Counter(all_characters)
characters, char_counts = zip(*character_counts.most_common(20))

# Plot character frequency
plt.figure(figsize=(12, 6))
sns.barplot(x=list(characters), y=list(char_counts))
plt.title('Top 20 Most Common Characters')
plt.xlabel('Characters')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("static/eda_plots/character_frequency.png")
plt.close()

# Unique words per intent
df['unique_words'] = df['pattern'].apply(lambda x: len(set(x.split())))
unique_words_by_intent = df.groupby('tag')['unique_words'].mean().sort_values()

plt.figure(figsize=(12, 6))
sns.barplot(x=unique_words_by_intent.index, y=unique_words_by_intent.values)
plt.title('Average Number of Unique Words per Intent')
plt.xlabel('Intent')
plt.ylabel('Average Unique Words')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("static/eda_plots/unique_words_per_intent.png")
plt.close()

# 5. Data Quality Checks
# Check for duplicate patterns
duplicates = df[df.duplicated(subset='pattern', keep=False)]
print("\nDuplicate Patterns:")
print(duplicates)

# Check for unique tags
unique_tags = df['tag'].nunique()
print(f"\nNumber of unique intents: {unique_tags}")
