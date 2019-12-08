### Topic modeling of arxiv scientific publications
This is an independent project that I took upon for fun. The goal of this project is to use the open publications on [arXiv.org](https://arxiv.org/) to build a recommender system for a query document. 

### Dataset 
The abstract, title, author names, date of publication, and arXiv category were downloaded using arXiv API from the year 1992 to 2018. Note that this large amount of data and the corresponding model output is not added as a part of this repository. 

There are two different subjects of publication that I will focus on: 
(1) Condensed Matter Physics, and (2) Machine Learning and Artificial Intelligence. 
For now, I am making two separate models for each subject. 

### Modeling (unsupervised)
The latent Dirichlet analysis (LDA) was used to identify (cluster) summarized topics in the corpus using only the title and abstract. As an output, each document has a topic assignment that is used as a position of the topic in K (number of topics) dimensional space to find the nearest neighbors of the document using the standard euclidean distance. 

### Outcome and Product
The end goal of this project is to build a web app that provides recommendations for an article based on the query article. This project can reduce the time that is spent looking for relevant scientific articles, allowing support for scientific innovation. 

### Disclaimer
The model was created as an independent fun project. Feel free to use it at your own risk. 
