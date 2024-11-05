from prompt.reason_after_sql import REASON_AFTER_SQL_PROMPT
from prompt.one_step import ONE_STEP_PROMPT
import pandas as pd
import re

def get_table_head(table, head = True)->str:
    if head:
        table = table.head(3).to_markdown(index=False)
    else:
        table = table.to_markdown(index=False)
    table = table.split('\n')
    ## remove index 1
    table.pop(1)
    table = '\n'.join(table)
    return table.lower()



def get_prompt(benchmark:str, type:str, data:list)->list:
    assert type in [ 'one_step', 'one_step_result']
    prompts = []
    for entry in data:
        if type == 'one_step':
            entry['question'] = entry['statement']
            entry['table_head'] = get_table_head(entry['table'], head = False)
            #print(entry.keys())
            prompt = ONE_STEP_PROMPT[benchmark].format(**entry)
            del entry['table']
            del entry['table_head']
            prompt = re.sub(' +', ' ', prompt)
        elif type == 'one_step_result':
            entry['question'] = entry['statement']
            entry['table_head'] = get_table_head(entry['table'], head = False)
            #print(entry.keys())
            try:
                prompt = REASON_AFTER_SQL_PROMPT[benchmark].format(**entry)
            except:
                entry['skip'] = True
            del entry['table']
            del entry['table_head']
            prompt = re.sub(' +', ' ', prompt)
        prompts.append(prompt)
    return prompts
if __name__ == '__main__':
    test_table = ''' Year | Competition | Venue | Position | Event | Notes\n2000 | World Junior Championships | Santiago, Chile | 1st | Discus throw | 59.51 m\n2003 | All-Africa Games | Abuja, Nigeria | 5th | Shot put | 17.76 m\n2003 | All-Africa Games | Abuja, Nigeria | 2nd | Discus throw | 62.86 m\n2004 | African Championships | Brazzaville, Republic of the Congo | 2nd | Discus throw | 63.50 m\n2004 | Olympic Games | Athens, Greece | 8th | Discus throw | 62.58 m\n2006 | Commonwealth Games | Melbourne, Australia | 7th | Shot put | 18.44 m\n2006 | Commonwealth Games | Melbourne, Australia | 4th | Discus throw | 60.99 m\n2007 | All-Africa Games | Algiers, Algeria | 3rd | Discus throw | 57.79 m\n2008 | African Championships | Addis Ababa, Ethiopia | 2nd | Discus throw | 56.98 m'''
    ## transform to pandas
    test_table = test_table.split('\n')
    test_table = [x.split('|') for x in test_table]
    test_table = pd.DataFrame(test_table[1:], columns = test_table[0])

    test_table = get_table_head(test_table)
    question = 'Q>?'
    title = 'TTTT'
    sql = 'SQL'
    reason = 'Reasoning'
    result = 'RESULT'
    print(ONE_STEP_PROMPT['wikitab'].format(table_head = test_table, question = question, title = title))
    print(REASON_AFTER_SQL_PROMPT['wikitab'].format(table_head = test_table, question = question, title = title, sql = sql, result = result, reason_before_sql = reason))
