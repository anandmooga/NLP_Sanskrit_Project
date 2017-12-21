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
FROM TempGraph
GROUP BY CNG_Group, Entity_1;


INSERT INTO Graph(Entity_1, Type_1, Entity_2, Type_2, Weight)
SELECT Entity_1, Type_1, CNG_Group, 'CNG_Group', SUM(Weight)
FROM TempGraph
GROUP BY CNG_Group, Entity_1;

--CNG_Group to CNG_Group edges are still left