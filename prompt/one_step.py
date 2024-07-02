ONE_STEP_PROMPT = {}

ONE_STEP_PROMPT['wikitab'] = '''This is a table question answering task. You will answer the question based on the given context.
You will answer the question based on the given context.You should reach a short-form answer after reasoning.
You are asked to answer the question in three steps.
1. Analyze the question and the given context. Make up a plan to answer the question.
2. Write one or more SQL to query the table for necessary information and output expected execution result.
3. Reason step-by-step to reach the final answer. You should write your final answer as <answer>.

/*
Question:
How many medals were won by Kenya?

Table title:
Athletics at the 2001 Goodwill Games 3000 meters

Table:
| Rank | Name              | Nationality    | Time    |
| 1    | Brahim Boulami    | Morocco        | 8:17.73 |
| 2    | Reuben Kosgei     | Kenya          | 8:18.63 |
| 3    | Stephen Cherono   | Kenya          | 8:19.98 |
| 4    | Bouabdellah Tahri | France         | 8:20.25 |
| 5    | Tim Broe          | United States  | 8:20.75 |
| 6    | Luis Miguel Martín| Spain          | 8:24.03 |
| 7    | Raymond Yator     | Kenya          | 8:27.19 |
| 8    | Thomas Chorny     | United States  | 9:24.26 |

Answer:
Let's answer in three steps.
1. Plan for answering the question
- The question focus on the medals won by Kenya. According to common sense, only top 3 players can win medals. 
- So we write a SQL to select the top three players. Then we count the numbers of medals won by Kenya based on the output of SQL.

2. Write SQL and execute SQL
SQL:
```sql
SELECT Name, Nationality FROM Table WHERE Rank <= 3 and Nationality = 'Kenya';
```

Executed result:
| Name            | Nationality |
| Reuben Kosgei   | Kenya       |
| Stephen Cherono | Kenya       |

3. Step-by-Step Reasoning
- The question is asking about the number of medals won by Kenya.

- Based on the execution result of SQL query, only two of the players that win medals are from Kenya.

- Therefore, the total number of medals won by Kenya is 2.

Final Answer:
<2>
*/

/*
Question:
{question}

Table Title:
{title}

Table:
{table_head}

Answer:'''

ONE_STEP_PROMPT['tabfact_small'] = '''This is a table fact verification task. You will predict whether the statement is true or false based on the given context.
You will answer the question based on the given context.You should reach a short-form answer after reasoning.
You are asked to answer the question in three steps.
1. Analyze the question and the given context. Make up a plan to answer the question.
2. Write one or more SQL to query the table for necessary information and output expected execution result.
3. Reason step-by-step to reach the final answer. You should write your final answer as <true> or <false>.

Statement:
Kenya won 3 medals in the 3000 meters event at the 2001 Goodwill Games.

Table title:
Athletics at the 2001 Goodwill Games 3000 meters

Table:
| Rank | Name              | Nationality    | Time    |
| 1    | Brahim Boulami    | Morocco        | 8:17.73 |
| 2    | Reuben Kosgei     | Kenya          | 8:18.63 |
| 3    | Stephen Cherono   | Kenya          | 8:19.98 |
| 4    | Bouabdellah Tahri | France         | 8:20.25 |
| 5    | Tim Broe          | United States  | 8:20.75 |
| 6    | Luis Miguel Martín| Spain          | 8:24.03 |
| 7    | Raymond Yator     | Kenya          | 8:27.19 |
| 8    | Thomas Chorny     | United States  | 9:24.26 |

Answer:
Let's answer in three steps.
1. Plan for answering the question
- The question focus on the medals won by Kenya. According to common sense, only top 3 players can win medals. 
- So we write a SQL to select the top three players. Then we count the numbers of medals won by Kenya based on the output of SQL.

2. Write SQL and execute SQL
SQL:
```sql
SELECT Name, Nationality FROM Table WHERE Rank <= 3 and Nationality = 'Kenya';
```

Executed result:
| Name            | Nationality |
| Reuben Kosgei   | Kenya       |
| Stephen Cherono | Kenya       |

3. Step-by-Step Reasoning
- The statement claims that Kenya won 3 medals in the 3000 meters event at the 2001 Goodwill Games.
- Based on the execution result of SQL query, only two of the players that win medals are from Kenya.
- Therefore, the table context does not support the statement. The statement is <false>.

Final Answer:
<false>
*/

/*
Statement:
{question}

Table Title:
{table_caption}

Table:
{table_head}

Answer:'''


ONE_STEP_PROMPT['feverous'] = '''This is a fact verification task. You are asked to check the veracity of the claim. Both table and sentence context are provided but you are not required to use both of them. You can use either of them or both of them.
You will predict the claim's label based on the given claim and context. There are 3 possible labels: <SUPPORTS>, <REFUTES>, <NOT ENOUGH INFO>. <SUPPORTS> means the claim is supported by the context. <REFUTES> means the claim is refuted by the context. <NOT ENOUGH INFO> means the context does not provide enough information to the support or refute the claim.
You are asked to check the claim in three steps.
1. Analyze the question and the given context. Make up a plan to verify the claim.
2. Write one or more SQL to query the table for necessary information and output expected execution result.
3. Reason step-by-step to reach the final answer.

/*
Claim:
Sony Dwi Kuncoro, born on July 7, 1984, was the champion of the Men's singles event of the 2009 Indonesia National Badminton Championship with a score of 21-17, 22-20. 

Table Context
Page Title: 2009 Indonesia National Badminton Championship
Caption:
Category | Winners | Runners-up | Score
Men's singles | Sony Dwi Kuncoro | Andre Kurniawan Tedjono | 21-17, 22-20
Women's singles | Maria Kristin Yulianti | Maria Febe Kusumastuti | 21-17, 17-21, 23-21
Men's doubles | Hendra Setiawan | Rian Sukmawan , Yonathan Suryatama Dasuki | 21-9, 21-13
Women's doubles | Meiliana Jauhari , Greysia Polii | Liliyana Natsir , Shendy Puspa Irawati | 21-19, 18-21, 21-17
Mixed doubles | Tontowi Ahmad , Greysia Polii | Muhammad Rijal , Debby Susanto | 21-11, 21-19".

Sentence Context
Sony Dwi Kuncoro (1): Sony Dwi Kuncoro (born 7 July 1984) is a badminton singles player from Indonesia.
Sony Dwi Kuncore (2): He reached a career high as world number 3 on 7 October 2004.
Sony Dwi Kuncore (3): In 2009, his best performance in Superseries was semifinalist in Indonesia Open, beaten by Taufik Hidayat, 17–21, 14–21 and in Denmark Open beaten by Marc Zwiebler of Germany in three tough games.

Answer:
Let's check the claim in 3 steps.\\\\
1. Plan for verify the claim
- Based on the claim and the given context. The information for winner of Men's singles can be obtained from the table. We could write a SQL to query the name of the winner and the score of the final for Men's singles.
- The claim also states that Sony Dwi Kuncoro is born on July 7, 1984. We will use the sentences to check this part of claim. 

2. Write SQL and execute SQL
SQL:
```sql
SELECT Winners FROM Table WHERE Category = Men's singles
```

Executed result:
Category | Winners  | Score
Men's singles | Sony Dwi Kuncoro | 21-17, 22-20 


3. Step-by-step Reasoning
- The claim states Sony Dwi Kuncoro, born on July 7, 1984, was the champion of the Men's singles event of the 2009 Indonesia National Badminton Championship with a score of 21-17, 22-20. 
- The SQL query result supports the claim. Sony Dwi Kuncoro was the champion of the Men's singles event of the 2009 Indonesia National Badminton Championship with a score of 21-17, 22-20.
- The sentence evidence Sony Dwi Kuncoro (1) supports that he was indeed born on July 7, 1984.
- The claim is supported by the context. Therefore, the answer is <SUPPORTS>.

Final Answer:
<SUPPORTS>
*/

/*
Claim:
{statement}

Table Context
Page Title: {page}
Caption: {caption}
Table:
{table_head}

Sentence Context
{sentences}

Answer:'''
