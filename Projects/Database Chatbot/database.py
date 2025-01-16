import sqlite3

## connect to sqllite
connection = sqlite3.connect("./student.db")

##create a cursor object to insert record,create table
cursor = connection.cursor()

## create the table
cursor.execute(
    """
create table STUDENT(NAME VARCHAR(25),CLASS VARCHAR(25),
SECTION VARCHAR(25),MARKS INT)
"""
)

## Insert records
cursor.execute("""Insert Into STUDENT values('Cypher','Data Science','A',90)""")
cursor.execute("""Insert Into STUDENT values('Deadlock','Data Science','B',100)""")
cursor.execute("""Insert Into STUDENT values('Sage','Data Science','A',86)""")
cursor.execute("""Insert Into STUDENT values('KJ','DEVOPS','A',50)""")
cursor.execute("""Insert Into STUDENT values('Gekko','DEVOPS','A',35)""")

## Display all the records
print("The inserted records are")
data = cursor.execute("""Select * from STUDENT""")
for row in data:
    print(row)

## Commit changes in the database
connection.commit()
connection.close()
