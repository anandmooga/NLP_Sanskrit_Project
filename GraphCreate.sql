
--creating a table called Extracted in which we are including the word as Lemma+CNG
DROP TABLE IF EXISTS Extracted;
CREATE TABLE Extracted( Filename INTEGER, Lemma TEXT, CNG INTEGER, Word TEXT, Index1 INTEGER, Index2 INTEGER);
--copy values from MyTable but add an extra column in which we concat Lemma, CNG into 1 string called word
INSERT INTO Extracted(Filename, Lemma, CNG, Word, Index1, Index2)
SELECT Filename, Lemma, CNG, Lemma || '_' || substr(CNG, -4, 4), Index1, Index2
FROM MyTable ;
DROP MyTable;

--To define realtions between Lemma, CNG and Word we create a Graph Database with weighted edges, nodes being CNG, Lemma and Word. We have 9 diffrent edges.
--This is possible is sql with innerjoins.
--Create a table called node which is a duplicate of Extracted, so that its possible to inner join

--Nodes 
DROP TABLE IF EXISTS Node;
CREATE TABLE Node(Lemma TEXT, CNG INTEGER, Word TEXT, Filename INTEGER);
INSERT INTO Node(Lemma, CNG, Word, Filename)
SELECT Lemma, CNG, Word, Filename
FROM Extracted;

--creating our graph ! 

--Graph table, Weight is defined as the co-occurance of an entity with other entities in a scentence.
DROP TABLE IF EXISTS Graph;
CREATE TABLE Graph( Entity_1 TEXT, Type_1 TEXT, Entity_2 TEXT, Type_2 TEXT, Weight INTEGER);

--Lemma Lemma Edge
INSERT INTO Graph( Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT e.Lemma, 'Lemma', n.Lemma, 'Lemma', COUNT(e.Filename)
FROM Extracted e, Node n
WHERE e.Filename = n.Filename AND e.rowid <> n.rowid
GROUP BY e.Lemma, n.Lemma;

--Lemma CNG Edge
INSERT INTO Graph( Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT e.Lemma, 'Lemma', substr(n.CNG, -4, 4), 'CNG', COUNT(e.Filename)
FROM Extracted e, Node n
WHERE e.Filename = n.Filename AND e.rowid <> n.rowid
GROUP BY e.Lemma, n.CNG;

--Lemma Word Edge
INSERT INTO Graph( Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT e.Lemma, 'Lemma', n.Word, 'Word', COUNT(e.Filename)
FROM Extracted e, Node n
WHERE e.Filename = n.Filename AND e.rowid <> n.rowid
GROUP BY e.Lemma, n.Word;

--CNG CNG Edge
INSERT INTO Graph( Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT substr(e.CNG, -4, 4), 'CNG', substr(n.CNG, -4, 4), 'CNG', COUNT(e.Filename)
FROM Extracted e, Node n
WHERE e.Filename = n.Filename AND e.rowid <> n.rowid
GROUP BY e.CNG, n.CNG;

--CNG Word Edge
INSERT INTO Graph( Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT substr(e.CNG, -4, 4), 'CNG', n.Word, 'Word', COUNT(e.Filename)
FROM Extracted e, Node n
WHERE e.Filename = n.Filename AND e.rowid <> n.rowid
GROUP BY e.CNG, n.Word;


--Word Word Edge
INSERT INTO Graph( Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT e.Word, 'Word', n.Word, 'Word', COUNT(e.Filename)
FROM Extracted e, Node n
WHERE e.Filename = n.Filename AND e.rowid <> n.rowid
GROUP BY e.Word, n.Word;


--CNG Lemma Edge
INSERT INTO Graph( Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT substr(e.CNG, -4, 4), 'CNG', n.Lemma, 'Lemma', COUNT(e.Filename)
FROM Extracted e, Node n
WHERE e.Filename = n.Filename AND e.rowid <> n.rowid
GROUP BY e.CNG, n.Lemma;

--Word CNG Edge
INSERT INTO Graph( Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT e.Word, 'Word', substr(n.CNG, -4, 4), 'CNG', COUNT(e.Filename)
FROM Extracted e, Node n
WHERE e.Filename = n.Filename AND e.rowid <> n.rowid
GROUP BY e.Word, n.CNG;

--Word Lemma Edge
INSERT INTO Graph( Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT e.Word, 'Word', n.Lemma, 'Lemma', COUNT(e.Filename)
FROM Extracted e, Node n
WHERE e.Filename = n.Filename AND e.rowid <> n.rowid
GROUP BY e.Word, n.Lemma;



