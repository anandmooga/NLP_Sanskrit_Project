This repository contains the code used for my project in Natural Langugae Processing in Sanskrit.

Aim: We are trying to create vector representaion for words using a graph, metapaths and mutual information regression. in our original pickle files we contain approximatly 4,40,000 pickle files each file containg a sentence its root word and morphological class  

Steps in the project:
1. Extraction of pickle files of class DCS. the pickle files can be downloaded from https://zenodo.org/record/803508 and can be extacted using the file Thread Extraction.ipynb
2. To create a graph with nodes as CNG, Lemma, and Word(Lemma+CNG). Can be made from Graph_Final.sql and can be cross checkd from Graph_Check.ipynb
3. To create methapaths with external nodes comprising of Lemma, CNG, Word and Internal nodes comprisng of distinct CNG's and CNG groups. Note: Verbgroups CNG's are -ve. Can be done using Metapahts.ipynb