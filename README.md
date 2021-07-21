# Distributional-Semantics-M

Files:
		dist_sim.py
		cbow_sim.py

	Output files:
		sim2_FREQ_output.txt
		sim_2_PMI_output.txt
		sim_10_PMI_output.txt
		sim_2_CBOW_output.txt

## Count-based Models of Distributional Semantic Similarity:
We create and evaluate models of Distributional Semantic Similarity.

Our system takes <window> <weighting>, where the window specifies the context window and weighting can be either:
Frequency (FREQ) or Positive Pointwise Mutual Information (PMI).

To create our vector representations:
We use the Brown Corpus and perform basic preprocessing, removing punctuation.
For each unique word in the corpus, we use a sorted list to create a dictionary with indices.
Going through the Brown Corpus with the specified window value, we count the occurrences of words within this context window.
We store these values as a sparse row matrix, with indices corresponding to the unique words in the corpus.

We compare our weights against a file with a pair of words and human judgments.
In the output, we return the ten highest weighted features for each word (in the word pair) along with our weighting.
We compute the cosine similarity of our words.
Then, we compute and print the Spearman Correlation between our scores and the human judgment scores.
A summary of our Spearman correlation is below.

Summary:
Window=2 Weighting=FREQ (sim_2_FREQ_output.txt)
	Correlation:0.131204319061681

Window=2 Weighting=PMI (sim_2_PMI_output.txt)
	Correlation:0.4586379716173717

Window=10 Weighting=PMI (sim_10_PMI_output.txt)
	Correlation:0.22931898580868584

With window=2, Weighting=FREQ, the ten highest weighted features includes frequent words that occur often around all words.
For example:
	car: is:9 to:13 was:14 with:18 of:23 his:23 in:24 and:26 a:57 the:140
	automobile: was:3 racing:4 by:4 to:4 in:5 state:5 an:5 and:6 of:8 the:20

Words such as 'the' or 'is' occur most frequently, and are considered among the highest weighted features.
However, these words tell us little about the semantic information of the words around them.

With window=2, Weighting=PMI, we attempt to control for these minimally informative words and our weights have a stronger correlation with those in the human judgment file.
With window=10, Weighting=PMI, the fall in correlation between our scores and the judgment file seems to indicate that the window may be too large.

Continuous Bag of Words (CBOW) using Word2Vec:
To build our predictive model, we used gensim with parameters:
	Word2Vec(sentences=corpus, window=window_val, size=35, min_count=1, workers=6, iter=100)

We attempted first the default gensim values of size=100, iter=5. However, the correlation score varied drastically.
We then modified the parameters until our scores stabilized, limiting the vector size and increasing iterations.

Summary:
Window=2 CBOW (sim_2_CBOW_output.txt)
	Correlation:0.4301654800907899

When evaluating our CBOW model, the model seems to capture some of the semantic similarities between word pairs.
For instance the positive similarity score for car,automobile:0.5153248310089111
And the negative score for rooster,voyage:-0.0022841463796794415
This is also reflected in the positive Spearman correlation.
