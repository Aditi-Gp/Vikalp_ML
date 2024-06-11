# Vikalp_ML
## Positive News Recommendation System
Dataset link: https://www.kaggle.com/datasets/hoshi7/news-sentiment-dataset?resource=download

**Problem Statement:**
Design a news algorithm which recommends positive/uplifting content to the users.

**Thought Process:**
Firstly, before proceeding with the project, I researched on the datasets available on the internet. I came across the dataset that contained news articles with sentiment labels (0 for negative and 1 for positive). After finding the ideal dataset, I drew a roadmap and chose the best preprocessing and model for this. 
   Proceeding  with the data preprocessing, I removed all the stop words, punctuation and converted alll text to lowercase. Then using TF-IDF, I converted the text data into sequences of integers. Further I trained the sentiment analysis model using machine learning algorithm so that the model should be able to classify news articles as positive or negative based on their sentiment. Then I evaluated the performance of the model using metrics like accuracy, precision, recall, and F1-score and fine-tuned the model by adjusting hyperparameters or using different algorithms to improve its performance.

 **Recommendation Algorithm:**
Once the sentiment analysis model was trained and evaluated, I used it to filter out negative news articles and ranked the remaining positive articles based on their relevance to the user's interests and preferences that recommended the top-ranked articles to the user.

**Challenges:**
Since the dataset contained more positive articles than negative ones, this could have led to biased model performance. Also, since I have not worked on recommendation algorithms, so I had to look for its implementation and working. Nextly, finding the dataset was a bit challenging too.

**Resolutions:**
I used techniques like oversampling the minority class (negative articles) or undersampling the majority class (positive articles) to balance the dataset and used class weights to adjust the model's performance based on the class imbalance. I used techniques like collaborative filtering  to incorporate user preferences into the recommendation algorithm. and techniques like matrix factorization or neural networks to model user preferences and item characteristics.

**Initial Compromises:**

1. **Data Selection:**
   - Use the provided dataset for training the sentiment analysis model.

2. **Model Selection:**
   - Use a simple machine learning algorithm like Naive Bayes or Logistic Regression for the initial model.

3. **Model Evaluation:**
   - Use a single evaluation metric like accuracy for the initial model evaluation.

4. **Recommendation Algorithm:**
   - Use a simple ranking algorithm like popularity-based ranking for the initial recommendation algorithm.

**Future Improvements:**

1. Collection of more diverse and balanced data to improve the model's performance.\

2. Incorporate user preferences and interests into the recommendation algorithm using techniques like collaborative filtering or content-based filtering.

# Classical music recommender system based on the time of day (Pahar) and the raga of the song.

### Problem Statement

The problem statement involves designing a classical music recommender system based on the time of day (Pahar) and the raga of the song. The goal is to predict the target variable based on the features provided in the dataset.

### Data Selection

For training the model, I used the following datasets:
1. **Train Data**: This dataset contained information about the songs, including the song ID, source system, source screen name, source type, and target to train the model.

2. **Songs Data**: This dataset contained information about the songs, including the song ID, song length, genre IDs, artist name, composer, lyricist, and language. This data has been used to enrich the features of the songs.

3. **Members Data**: This dataset contained information about the members, including the msno, city, birth date, gender, registered via, and registration and expiration dates. This data has been used to enrich the features of the members.

### Challenges
I initally faced the challenge of finding the dataset for Classical

1. **Handling Missing Values**: The datasets contain missing values, which need to be handled to ensure the model is robust.

2. **Feature Engineering**: The datasets require feature engineering to extract relevant information that can be used for training the model.

3. **Handling Categorical Variables**: The datasets contain categorical variables, which need to be handled using techniques such as one-hot encoding or label encoding.

4. **Handling Imbalanced Data**: The datasets may contain imbalanced data, which can affect the performance of the model. Techniques such as oversampling the minority class or undersampling the majority class can be used to address this.

5. **Handling High-Dimensional Data**: The datasets contain high-dimensional data, which can lead to the curse of dimensionality. Techniques such as dimensionality reduction can be used to address this.

### Solution

1. **Handling Missing Values**: We will use the `reduce_mem_usage` function to handle missing values by converting them to the appropriate data type.

2. **Feature Engineering**: We will use techniques such as splitting the genre IDs into separate columns and converting the language to a numerical value.

3. **Handling Categorical Variables**: We will use label encoding to handle categorical variables.

4. **Handling Imbalanced Data**: We will use oversampling the minority class to address imbalanced data.

5. **Handling High-Dimensional Data**: We will use dimensionality reduction techniques such as PCA to address high-dimensional data.

### Initial Compromises

1. **Feature Selection**: We will select the most relevant features based on their correlation with the target variable.

2. **Model Selection**: We will use a random forest classifier as the initial model, which can handle high-dimensional data and categorical variables.

3. **Hyperparameter Tuning**: We will use grid search to tune the hyperparameters of the model.

### Future Improvements

1. Use advanced techniques such as deep learning models or graph-based models to improve the performance of the model.

2. **Use of Additional Features**: We can use additional features such as the lyrics of the songs or the mood of the songs to improve the performance of the model.

3. **Use of Transfer Learning**: We can use transfer learning to leverage pre-trained models and fine-tune them for our specific problem.

4. **Use of Ensemble Methods**: We can use ensemble methods such as bagging or boosting to improve the performance of the model.

5. **Use of Hyperparameter Tuning**: We can use more advanced hyperparameter tuning techniques such as Bayesian optimization to further improve the performance of the model.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/20029510/21a02b1e-b88f-4f54-b9ab-b4bd120aad23/paste.txt
