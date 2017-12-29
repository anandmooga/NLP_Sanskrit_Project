

This repository contains the code used for my project in Natural Language Processing in Sanskrit.

Aim: We are trying to create vector representation for sanskrit words using concept of Graphs. And further use the vectors for various regression/classification/clustering problems.

Language : Python and Sqlite

https://zenodo.org/record/803508

The above link contains the files which we will use

Download:

This repository contains the code used for my project in Natural Language Processing in Sanskrit.

Aim: We are trying to create vector representation for sanskrit words using concept of Graphs. And further use the vectors for various regression/classification/clustering problems.

Language : Python and Sqlite

https://zenodo.org/record/803508

The above link contains the files which we will use

Download:

- DCS\_pick.zip : Contains 442383 files, but there are around 2000         missing files
- pickleReader.py: Contains code to read pickles
- DCS\_999.p : it is an example pickle file, can be extracted using pickleReader.py
- DCS: Digital Corpus of Sanskrit

First objective is to extract all of the Lemma, CNG from each sentence.

- The file ThreadExtraction.ipynb contains code for doing this.
- We store all of these values in a single dataframe called         Extracted.db in a table called MyTable.
- MyTable contains 32,43,895 entries and 5 columns.
- Columns Filename, CNG and Lemma are self-explanatory.
- Index1 corresponds to the position of word in the sentence.
- Index 2 corresponds to the position of a Lemma/CNG in a word.

A word in sanskrit can contain 1 or more Lemma/Root word. A CNG is a morphological class which in our case is Verb or Noun. Verb are represented by negative CNG values.

In order to extract maximum information from any word we try to represent it in the form of a Graph.

- We consider 3 nodes in our graph: Lemma, CNG and word(Lemma+CNG).
- Note: this word is different the the word in a sentence.
- There are 9 different possible edges, as we have 3 nodes.
- Edges have weights
- Weights are defined as : The count of co-occurrence of an edge with another edge in the same file. But they should not be in the same row as that information is already captured in word.
- Example : Let&#39;s take an example of a lemma co-occurring with a cng , so in a sentence if a lemma and cng exist and index1 and index 2 of lemma and CNG are different they are said to co-occur in that sentence. similarly rest of the co-occurrences can be defined.
- The file GraphCreate.sql is used for this.
- In order to cross-check the results obtained by using sql, we used the brute force method to create the graph in python, and then compared the results.
- Code for bruteforce method is in GraphCheck.ipynb

A CNG corresponds to either a noun or a verb. A CNG Group is a group which contains CNG&#39;s of similar nature. A dictionary of all CNG Groups can be found in noungroups.json and verbgroups.json (note: in verbgroups the CNG&#39;s are actually negative, but they are stored as positive integers, so we have to make the change while using it)

We want to include information of CNG\_Groups in the graph as well so we add an additional node CNG\_group. And respectively 7 additional edges.

- First we extract information from the json files and store it in the database.
- The file CNGGroups.ipynb contains code for the above
- The next step is to modify our graph and include CNG\_Group as a Node, and define 7 new edges.
- Weights of the edges of CNG\_Groups with an Entity are defined as the sum of weights of all the edges with CNG&#39;s of that CNG\_group with the Entity.
- Example: lemma1-&gt;CNG1 has weight 6 and lemma1-&gt;CNG4 has weight 2. and CNG\_Group1 consists of CNG1 and CNG4. weight of edge         lemma1-&gt;CNG\_group1 will be 8(since 6+2)
- GraphUpdate.sql is used for this
- There are a total of 6,77,62,647 entries

In order to establish relationships between different words we sample sub-graphs from our main graph. We call them Metapaths. We create metapaths of size 2,3 and 4. as if we go beyond this computation is too expensive.

A metapath can consist any nodes in its path but, in order to reduce computation we use metapths with the most information. We use a metapath with External nodes as and Lemma, CNG or Word but internal nodes only as CNG&#39;s or CNG\_Groups as they are the Morphological class and contain information on relation between words in a sentence.

So External Nodes are : Lemma, CNG and Word and Internal Nodes can be any of the distinct CNG&#39;s or CNG\_Groups.

We will first build the skeleton of the Metapaths and then we will substitute some sample values to get the most useful metapaths.

To build the skeleton of the Metapaths :

- We use the file MetapathsGeneration.ipynb to generate the metapaths
- We use strings &#39;Lemma&#39;, &#39;CNG&#39;, &#39;Word&#39; as External nodes         so that when we substitute values in metapath skeletons they are easy         to replace.
- Metapath2 has 9 types
- Metapath3 has 4,204 types
- Metapath4 has 19,62,802 types

We want to use Metapaths as features to show relation between 2 words in a sentence, as it can be used recursively.

So we sample 10,000 unique word word edges with their weights and 1000 word word edges with 0 weight, from which we will calculate the probability of going from externalnode1 to externalnode2 via all the Skeleton metapaths we defined. i.e. 11,000 rows \* (9+4204+1962802)columns.

A word contains both lemma and CNG so we get all the information we need to follow the metapaths we defined as external nodes could be Lemma,CNG or Word.

Probability of using a Metapaths is determined by Bigram Probability.

P(A,B) = Co-occurance(A,B)/Count(A).

Count(A) = Number of distinct sentences in which A occurs

Example:

if metapath is Word&gt;CNG4-&gt;CNG\_group7-&gt;Lemma

And word word pair is say, Lakshmanath Ramasya

Score/Probability = P(Lakshmanath,CNG4)\*P(CNG4, CNG\_group7)\*P(CNG\_group7, Rama)

where Rama is Lemma for Ramasya

Our labels for a sample will be the Metapath2 between Word and Word.

We do this in a series of steps as follows:

- We create a table with only word word edges so that we can sample 10,000 values from it
- The file GraphUpdate.sql contains code for this
- WordGraph contains 16069844 entries
- Take a sample of 10,000 pairs of word word edges, while making sure we get a good representative sample
- this is done by Sampling.ipynb
- The next step is to sample 1,000 word -&gt;word edges with zero weights and we make sure that the words in these 1,000 samples are from the 10,000 samples we selected before otherwise these samples will be meaningless
- We do this using the code in Sampling.ipynb &#39;s end
- In order to calculate bigram probability we need to query the weight of the edge from the database every time which is time consuming, so to speed up computation we same the entire graph in a dictionary.
- This is done by GraphDictonary.py
- To calculate Bigram probability/score we have the numerator but we need to denominators, which is count.
- We find the denominators by using Denominators.ipynb
- Then we find the score for 11,000 samples in each metapath. Which are our features
- This is done using FeaturesMetapath.ipynb

Now from (9+4204+1962802) metapaths we have we select 1 Lakh metapaths as by finding the number of unique scores for a metapath and selecting the metapaths with maximum unique values

Further we reduce the size to 20,000 by finding correlations of each vector of size 11,000 of metapaths with our label i.e. word-&gt;word metapath

Then we use mutual information regression to extract the best metapaths from the 20,000


