import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

file_path = "/Users/irisbao/Desktop/AI-blended questions/学生日志-ENG/AI-blended questions-sum.xlsx"
df = pd.read_excel(file_path)

print(df.head())

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{10}\b', '', text)
    text = text.replace('　', ' ').replace('０', '0').replace('１', '1').replace('２', '2').replace('３', '3').replace('４', '4').replace('５', '5').replace('６', '6').replace('７', '7').replace('８', '8').replace('９', '9').replace('Ａ', 'A').replace('Ｂ', 'B').replace('Ｃ', 'C').replace('Ｄ', 'D').replace('Ｅ', 'E').replace('Ｆ', 'F').replace('Ｇ', 'G').replace('Ｈ', 'H').replace('Ｉ', 'I').replace('Ｊ', 'J').replace('Ｋ', 'K').replace('Ｌ', 'L').replace('Ｍ', 'M').replace('Ｎ', 'N').replace('Ｏ', 'O').replace('Ｐ', 'P').replace('Ｑ', 'Q').replace('Ｒ', 'R').replace('Ｓ', 'S').replace('Ｔ', 'T').replace('Ｕ', 'U').replace('Ｖ', 'V').replace('Ｗ', 'W').replace('Ｘ', 'X').replace('Ｙ', 'Y').replace('Ｚ', 'Z').replace('ａ', 'a').replace('ｂ', 'b').replace('ｃ', 'c').replace('ｄ', 'd').replace('ｅ', 'e').replace('ｆ', 'f').replace('ｇ', 'g').replace('ｈ', 'h').replace('ｉ', 'i').replace('ｊ', 'j').replace('ｋ', 'k').replace('ｌ', 'l').replace('ｍ', 'm').replace('ｎ', 'n').replace('ｏ', 'o').replace('ｐ', 'p').replace('ｑ', 'q').replace('ｒ', 'r').replace('ｓ', 's').replace('ｔ', 't').replace('ｕ', 'u').replace('ｖ', 'v').replace('ｗ', 'w').replace('ｘ', 'x').replace('ｙ', 'y').replace('ｚ', 'z')
    return text

df['reflection_cleaned'] = df['reflection'].apply(clean_text)

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

vocab_mapping = {
    "doubao": "ai",
    "welearn": "ai",
    "preview": "preparation",
    "online": "ai",
    "offline": "classroom"}


def get_wordnet_pos(treebank_tag):
    tag = treebank_tag[0].upper() if treebank_tag else 'N'
    tag_dict = {"J": 'a', "V": 'v', "R": 'r', "N": 'n'}
    return tag_dict.get(tag, 'n') 
    
def preprocess_text(text):
    cleaned = clean_text(text)
    lower_text = cleaned.lower()
    tokens = word_tokenize(lower_text)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if token.isalnum()]
    tokens = [token for token in tokens if token not in stop_words]

    pos_tags = pos_tag(tokens)
    lemmas = []
    for word, tag in pos_tags:
        pos = get_wordnet_pos(tag)
        lemma = lemmatizer.lemmatize(word, pos=pos)
        lemmas.append(lemma)

    processed = [vocab_mapping.get(lemma, lemma) for lemma in lemmas]

    return ' '.join(processed)

remove_words = {"didnt", "dont", "also", "doesnt", "cant", "couldnt", "couldnt"}

df['reflection_preprocessed'] = df['reflection_cleaned'].apply(preprocess_text)

test_case = "Previewing texts made me more confident during the class."
processed = preprocess_text(test_case)
print(f"输入：{test_case}")
print(f"输出：{processed}") 

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0, metric='cosine')
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', prediction_data=True)
vectorizer_model = CountVectorizer(stop_words=stop_words)
ctfidf_model = ClassTfidfTransformer()

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    ctfidf_model=ctfidf_model,
    top_n_words=10
)

topics, probs = topic_model.fit_transform(df['reflection_preprocessed'])

topic_info = topic_model.get_topic_info()
print("主题信息：", topic_info)
topic_info.to_csv("topic_info.csv", index=False)
print("主题信息已保存到topic_info.csv")

df['topic'] = topics
print("文档主题分布如下：")
print(df[['reflection', 'topic']])

all_topics = topic_model.get_topics()

for topic_id, topic_info in all_topics.items():
    # topic_info是一个列表，每个元素是(关键词, 重要度)
    keywords = [word for word, _ in topic_info]
    importance = [imp for _, imp in topic_info]

    topic_df = pd.DataFrame({
        '关键词': keywords,
        '重要度': importance
    })
    topic_df.to_csv(f"topic_{topic_id}_keywords.csv", index=False)
    print(f"主题{topic_id}的关键词及其重要度已保存到topic_{topic_id}_keywords.csv")

if len(probs.shape) == 1:
    probs = probs.reshape(-1, 1)

print("probs的形状：", probs.shape)

num_topics = len(all_topics) 

for topic_id in range(num_topics):
    if topic_id < probs.shape[1]:
        topic_probs = probs[:, topic_id]
    else:
        topic_probs = np.zeros(probs.shape[0])

    df_topic = pd.DataFrame({
        '文档': df['reflection'],
        '主题概率': topic_probs
    })
    df_topic.to_csv(f"topic_{topic_id}_prob.csv", index=False)
    print(f"主题{topic_id}的概率分布已保存到topic_{topic_id}_prob.csv")

keyword_importance = []
topics = topic_model.get_topics()  

for topic_id, topic_info in topics.items():
    keywords, importance = zip(*topic_info)
    top_keywords = keywords[:20]
    top_importance = importance[:20]
    keyword_importance.append({
        '主题ID': topic_id,
        '关键词': ', '.join(top_keywords),
        '重要度': ', '.join([f'{i:.4f}' for i in top_importance])
    })

keyword_df = pd.DataFrame(keyword_importance)
keyword_df.to_csv("topic_keyword_importance.csv", index=False)
print("关键词重要度计算结果已保存到topic_keyword_importance.csv")

def generate_wordclouds(all_topics):
    for topic_id, topic_info in all_topics.items():
        keywords = [word for word, _ in topic_info]
        importance = [imp for _, imp in topic_info]

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(dict(zip(keywords, importance)))
    
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f'主题 {topic_id} 的词云图')
        plt.axis('off')
        plt.savefig(f'wordcloud_topic_{topic_id}.png')
        plt.close()
        print(f'主题 {topic_id} 的词云图已保存为 wordcloud_topic_{topic_id}.png')

generate_wordclouds(all_topics)


def plot_topic_word_distribution(all_topics):
    for topic_id, topic_info in all_topics.items():
        keywords = [word for word, _ in topic_info]
        importance = [imp for _, imp in topic_info]
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=keywords, palette='viridis')
        plt.title(f'主题 {topic_id} 的关键词重要度分布')
        plt.xlabel('重要度')
        plt.ylabel('关键词')
        plt.savefig(f'topic_word_distribution_{topic_id}.png')
        plt.close()
        print(f'主题 {topic_id} 的关键词重要度分布图已保存为 topic_word_distribution_{topic_id}.png')

plot_topic_word_distribution(all_topics)


def visualize_document_distribution(df, topics):
   
    if len(topics.shape) == 1:
        topics = topics.reshape(-1, 1)
    
    reducer = UMAP(random_state=42)
    embeddings = reducer.fit_transform(topics)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=embeddings[:, 0], y=embeddings[:, 1], hue=df['topic'], palette='tab20', legend='full')
    plt.title('文档主题分布图')
    plt.savefig('document_distribution.png')
    plt.close()
    print('文档主题分布图已保存为 document_distribution.png')

visualize_document_distribution(df, probs)


def visualize_inter_topic_map(all_topics):
   
    topic_embeddings = []
    for topic_info in all_topics.values():
        keywords, importance = zip(*topic_info)
      
        topic_embeddings.append(importance)
    topic_embeddings = np.array(topic_embeddings)
    
    similarity = cosine_similarity(topic_embeddings)
    
    reducer = UMAP(random_state=42)
    topic_positions = reducer.fit_transform(similarity)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=topic_positions[:, 0], y=topic_positions[:, 1], hue=range(len(topic_positions)), palette='tab20', legend='full')
    plt.title('隐含主题分布图')
    plt.savefig('inter_topic_map.png')
    plt.close()
    print('隐含主题分布图已保存为 inter_topic_map.png')

visualize_inter_topic_map(all_topics)


def plot_topic_heatmap(all_topics):
    
    topic_embeddings = []
    for topic_info in all_topics.values():
        keywords, importance = zip(*topic_info)
        topic_embeddings.append(importance)
    topic_embeddings = np.array(topic_embeddings)
    # 计算余弦相似度
    similarity = cosine_similarity(topic_embeddings)
    # 绘制热图
    plt.figure(figsize=(10, 10))
    sns.heatmap(similarity, annot=True, cmap='viridis', xticklabels=range(len(all_topics)), yticklabels=range(len(all_topics)))
    plt.title('主题间相似度热图')
    plt.savefig('topic_similarity_heatmap.png')
    plt.close()
    print('主题间相似度热图已保存为 topic_similarity_heatmap.png')

plot_topic_heatmap(all_topics)


def plot_hierarchical_clustering(all_topics):
    
    topic_embeddings = []
    for topic_info in all_topics.values():
        keywords, importance = zip(*topic_info)
        topic_embeddings.append(importance)
    topic_embeddings = np.array(topic_embeddings)
   
    similarity = cosine_similarity(topic_embeddings)
   
    distance_matrix = 1 - similarity
 
    linkage_matrix = linkage(distance_matrix, method='ward')
    
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix, labels=range(len(all_topics)), orientation='top', distance_sort='descending', show_leaf_counts=True)
    plt.title('层次聚类树状图')
    plt.savefig('hierarchical_clustering.png')
    plt.close()
    print('层次聚类树状图已保存为 hierarchical_clustering.png')

plot_hierarchical_clustering(all_topics)




