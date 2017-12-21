--Creating a table with only word -> word edges so that we can sample 10,000 External node in our metapath , and we use only word as they contain information of both CNG and Lemma
DROP TABLE IF EXISTS Wordgraph;
CREATE TABLE Wordgraph( Entity_1 TEXT, Type_1 TEXT, Entity_2 TEXT, Type_2 TEXT, Weight);

INSERT INTO Wordgraph( Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT e.Word, 'Word', n.Word, 'Word', COUNT(e.Filename)
FROM Extracted e, Node n
WHERE e.Filename = n.Filename AND e.rowid <> n.rowid
GROUP BY e.Word, n.Word;

--Creating CNG_groups Edges

DROP TABLE IF EXISTS Tempgraph;
CREATE TABLE Tempgraph(Entity_1 TEXT, Type_1 TEXT, Entity_2 TEXT, Type_2 TEXT, Weight INTEGER, CNG_Group TEXT);

INSERT INTO Tempgraph(Entity_1, Type_1, Entity_2, Type_2, Weight, CNG_Group)
SELECT t1.Entity_1, t1.Type_1, t1.Entity_2, t1.Type_2, t1.Weight, t2.CNG_Group
FROM Graph t1, 'CNG&CNGG' t2
WHERE t1.Entity_2 = t2.CNG ;

--Inserting our values into graph 

INSERT INTO Graph(Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT CNG_Group, 'CNG_Group', Entity_1, Type_1, SUM(Weight)
FROM Tempgraph
GROUP BY CNG_Group, Entity_1;


INSERT INTO Graph(Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT Entity_1, Type_1, CNG_Group, 'CNG_Group', SUM(Weight)
FROM Tempgraph
GROUP BY CNG_Group, Entity_1;

--CNG_Group to CNG_Group edges are still left
--We will duplicate the values of CNG_Group in TempGraph to another column in the same table,  so that we can group by 

DROP TABLE IF EXISTS Tempgraph;
CREATE TABLE Tempgraph(Entity_1 TEXT, Type_1 TEXT, Entity_2 TEXT, Type_2 TEXT, Weight INTEGER, CNG_Group1 TEXT, CNG_Group2 TEXT);

INSERT INTO Tempgraph(Entity_1, Type_1, Entity_2, Type_2, Weight, CNG_Group1, CNG_Group2)
SELECT t1.Entity_1, t1.Type_1, t1.Entity_2, t1.Type_2, t1.Weight, t2.CNG_Group, t3.CNG_Group
FROM Graph t1, 'CNG&CNGG' t2, 'CNG&CNGG' t3
WHERE t1.Entity_1 = t2.CNG AND t1.Entity_2 = t3.CNG;

INSERT INTO Graph(Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT CNG_Group1, 'CNG_Group', CNG_Group2, 'CNG_Group', SUM(Weight)
FROM Tempgraph
WHERE Type_1 = 'CNG' AND Type_2 = 'CNG'
GROUP BY CNG_Group1, CNG_Group2;
