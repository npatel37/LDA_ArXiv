### Topic modeling of arxiv scientific publications
This is an independent project that I took upon for fun. The goal of this project is to use the open publications on [arXiv.org](https://arxiv.org/) to build a recommender system for a query document. 

### Dataset 
The abstract, title, author names, date of publication, and arXiv category were downloaded using arXiv API from the year 1992 to 2018. The overall dataset contains approximately 224607 documents. Note that this large amount of data and the corresponding model is not added as a part of this repository. 

### Modeling (unsupervised)
The latent Dirichlet analysis (LDA) was used to identify (cluster) summarized topics in the corpus using only the title and abstract. As an output, each document has a topic assignment that is used as a position of the topic in K (number of topics) dimensional space to find the nearest neighbors of the document using the standard euclidean distance. 
Note that because of the large (6-8 hours) training time, finding the optimal number of the topic has not been performed currently. 

### Outcome and Product
This machine learning model is developed to build a web app that provides recommendations for an article based on the current article. This project can reduce the time that is spent looking for relevant scientific articles, allowing support for scientific innovation. 

### Disclaimer
The model was created as an independent fun project. Feel free to use it at your own risk. 
