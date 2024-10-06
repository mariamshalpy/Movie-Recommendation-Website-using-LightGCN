# Movies Recommendation Website Using LightGCN
<p align="center">
  <img src="https://github.com/user-attachments/assets/8711c32c-e5cb-45aa-9bdb-a8b5fcc29784" height="300"/>
</p>

In recent years, machine learning has become integral to marketing, with Graph Neural Networks (GNNs) emerging as a key technique, especially in Recommender Systems. This project focuses on implementing the LightGCN model, which utilizes user-item interaction graphs to improve movie recommendations. By treating movies as nodes and user ratings as edges, LightGCN learns user and item embeddings through linear propagation, capturing complex relationships for more accurate suggestions.

The methodology involves comparing LightGCN with five other collaborative filtering methods: Matrix Factorization (MF), Neural Graph Collaborative Filtering (NGCF), Cosine Similarity, Singular Value Decomposition (SVD), and a Hybrid Recommendation System. Performance is evaluated using precision and recall metrics to recommend top movies, utilizing publicly available datasets.

The project aims to build a user-friendly recommendation website, leveraging HTML, CSS, JavaScript for the front-end, and Flask for the back-end. LightGCN achieved a notable 96% accuracy, enhancing user experience through an interactive interface. Recommender Systems are crucial across various sectors, providing personalized suggestions that improve engagement and drive business success.

# Frontend
In frontent we have 2 folders here consisting of Templates folder where html files are present and Static folder where css_files and jss_files are present for the website front.
### Templates Folder
1. **home.html:** The introductory page featuring buttons for Sign Up and Login.
2. **index.html:** The main interface with all essential functionalities.

### Static Folder
This folder holds the static assets for the website.
1. **CSS Files:** Stylesheets that define the visual appearance of the website.
2. **JavaScript** Files: Scripts that add interactivity and dynamic functionality to the web pages.

# Proposed-Methodology-LightGCN
The proposed method in this project is LightGCN, or Light Graph Convolutional Network, is an innovative recommendation algorithm designed to provide accurate and personalized recommendations in various domains, particularly in collaborative filtering tasks. It leverages graph neural networks (GNNs) to analyze and model user-item interaction data, capturing latent relationships between users and items within a recommendation graph.
Process:
1. Data Processing
2. LightGCN Implementation
3. Evaluation with other 5 Collaborative filtering Models
   (MF, NGCF, Cosine similarity, SVD, Hybrid Recommendation systems)
5. Website Development
6. Integration
## Layers in LightGCN:
**a) Input Layer (Graph Representation):**
1. Users and items are represented as nodes in the graph & Interactions between users and items are represented as edges in the graph.
2. Each edge carries information about the interaction type or strength (e.g., ratings, genre, etc).
3. The graph can be directed or undirected depending on the nature of the interactions.

**b) Embedded Layer:**
1. Node embeddings are initialized for both users and items.
2. Each node (user or item) is associated with an initial embedding vector, These embeddings serve as the starting point for information propagation.
   
**c) Propagation Layer:**
1. Graph convolutional layers propagate information across the graph.
2. Information from neighboring nodes is aggregated to update node embeddings.
3. LightGCN performs this aggregation without feature transformations or nonlinear activations, making the process efficient,

**d) Prediction Layer:**
1. The final layer aggregates embeddings from the last convolutional layer.
2. Predictions for user-item interactions are generated based on these aggregated embeddings.
3. Common methods include computing the dot product or cosine similarity between user and item embeddings.

# The-Algorithmns-compared-with-LightGCN-algorithm
Algorithms Used For Comparison With LightGCN;
a) Hybrid Recommendation Systems:
b) Cosine Similarity Based Recommendation Systems
c) Neural Graph Collaborative Filtering Recommendation System (NGCF)
d) Matrix Factorisation (MF)
e) Singular Vector Decomposition (SVD)

**a) Drawbacks Of Hybrid Recommendation Systems:**

1. Simplicity and Efficiency: LightGCN is easier to implement and more computationally efficient than hybrid systems, which often require the integration of multiple models and techniques.
2. Cold-Start Problem Handling: LightGCN excels at addressing the cold-start problem by relying solely on user-item interaction data, making it more effective in scenarios with limited historical data for new items or users.

**b) Drawbacks Of Cosine Similarity Based Recommendation Systems:**
1. Implicit Feedback Handling: It's great at understanding scenarios where users don't explicitly rate items. This means it can pick up on subtle relationships between users and items better than methods like cosine similarity.
2. Embedding Learning: LightGCN learns compact, rich representations of users and items, Unlike cosine similarity, which just looks at basic features, LightGCN digs deeper, capturing complex relationships in the data. This leads to more accurate recommendations

**c) Drawbacks Of Neural Graph Collaborative Filtering:**
1. Complexity: Neural Graph Collaborative Filtering (NGCF) can be more complex to implement and fine-tune compared to LightGCN, which may require more expertise and computational resources.
2. Interpretability: NGCF's intricate neural network architecture may lack interpretability compared to LightGCN. making it harder to understand how are generated.

**d) Drawbacks Of Matrix Factorisation:**
1. Scalability: LightGCN handles large-scale datasets more efficiently. It can process massive amounts of data quickly, making it suitable for real-world recommendation systems with millions of users and items,
2. Cold-start Problem; LightGCN tackles the issue of making recommendations for new users or items with limited data, It can use additional information, like user or item characteristics, to provide meaningful recommendations even when interaction data is sparse.

**e) Drawbacks Of Singular Vector Decomposition:**
1. Cold Start Problem: SVD struggles with new users or items without enough interaction data, limiting its effectiveness in recommending to them. LightGCN handles this better by not relying. solely on historical interactions,
2. Scalability: SVD's computational demands grow quickly with dataset size, making it inefficient for large-scale recommendation tasks. LightGCN is more scalable and efficient due to its simpler operations and lower computational requirements

<p align="center">
  <img src="https://github.com/user-attachments/assets/a3194540-8c36-4c43-9e97-e1d5040af8fd" height="300"/>
</p>

# Dataset-ML-Latest-Small
This dataset (ml-latest-small) describes 5-star rating and free-text tagging activity from [MovieLens](http://movielens.org), a movie recommendation service. It contains 100836 ratings and 3683 tag applications across 9742 movies. These data were created by 610 users between March 29, 1996 and September 24, 2018. This dataset was generated on September 26, 2018.
Users were selected at random for inclusion. All selected users had rated at least 20 movies. No demographic information is included. Each user is represented by an id, and no other information is provided.
This and other GroupLens data sets are publicly available for download at <http://grouplens.org/datasets/>, making it a valuable resource for evaluating recommendation system methods.
DATASET LINK: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

The data are contained in the files are: "'links.csv', 'movies.csv', 'ratings.csv' and 'tags.csv" ".
Certainly! Here's a brief summary of each file in the MovieLens dataset:
1. **movies.csv:**
   - Contains movie information.
   - Each line represents one movie and includes movie ID, title, and genres.
   - Genres are listed in a pipe-separated format, and titles include the year of release in parentheses.  
2. **tags.csv:**
   - Contains user-generated tags for movies.
   - Each line represents one tag applied to one movie by one user.
   - Tags are typically single words or short phrases describing the movie.
   
3. **links.csv:**
   - Provides identifiers to link to other sources of movie data.
   - Each line represents one movie and includes movie ID, IMDb ID, and TMDb ID.
   - IMDb and TMDb IDs are identifiers for movies used by IMDb and TMDb websites, respectively.

4. **ratings.csv:**
   - Contains user ratings for movies.
   - Each line represents one rating of one movie by one user.
   - Ratings are made on a 5-star scale with half-star increments, and timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

These files collectively provide comprehensive data for analyzing user interactions and movie characteristics within the MovieLens dataset.
