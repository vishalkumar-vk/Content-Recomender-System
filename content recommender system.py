#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


credits = pd.read_csv("tmdb_5000_credits.csv")
movies_df = pd.read_csv("tmdb_5000_movies.csv")


# In[3]:


credits.head()


# In[5]:


movies_df.head()


# In[8]:


print("Credits:",credits.shape)
print("Movies Dataframe:",movies_df.shape)


# In[9]:


credits_column_renamed = credits.rename(index=str, columns={"movie_id":"id"})
movies_df_merge = movies_df.merge(credits_column_renamed, on='id')
movies_df_merge.head() 


# In[11]:


movies_cleaned_df = movies_df_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
movies_cleaned_df.head()


# In[12]:


movies_cleaned_df.info()


# In[13]:


movies_cleaned_df.head(1)['overview']


# In[14]:


from sklearn. feature_extraction. text import TfidfVectorizer
tfv = TfidfVectorizer (min_df=3, max_features=None,
strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
ngram_range= (1, 3),
stop_words='english')
# Filling NaNs with empty string
movies_cleaned_df [ 'overview'] = movies_cleaned_df [ 'overview'].fillna('')


# In[15]:


# Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform (movies_cleaned_df [ 'overview'])


# In[16]:


tfv_matrix


# In[17]:


tfv_matrix.shape


# In[20]:


from sklearn.metrics.pairwise import sigmoid_kernel
# Compute the sigmoid kernel
sig = sigmoid_kernel(tfv_matrix, tfv_matrix)


# In[21]:


sig[0]


# In[23]:


# Reverse mapping of indices and movie titles
indices= pd.Series(movies_cleaned_df.index, index=movies_cleaned_df['original_title']).drop_duplicates ()


# In[24]:


indices


# In[25]:


indices['Newlyweds']


# In[26]:


sig[4799]


# In[27]:


list(enumerate(sig[indices['Newlyweds']]))


# In[30]:


sorted(list(enumerate(sig[indices ['Newlyweds']])), key=lambda x: x[1], reverse=True)


# In[34]:


def give_rec(title, sig=sig):
    # Get the index corresponding to original_title
    idx = indices [title]
    # Get the pairwsie similarity scores
    sig_scores = list(enumerate (sig[idx]))
    # Sort the movies
    sig_scores = sorted (sig_scores, key=lambda x: x[1], reverse=True)
    # Scores of the 10 most similar movies
    sig_scores = sig_scores[1:11]
    # Movie indices
    movie_indices = [i[0] for i in sig_scores]
    # Top 10 most similar movies
    return movies_cleaned_df ['original_title'].iloc [movie_indices]


# In[39]:


# Testing our content-based recommendation system with the seminal film Spy Kids
give_rec ('Spy Kids')


# In[ ]:




