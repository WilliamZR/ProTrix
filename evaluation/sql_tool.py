import re
import pandas as pd
import sqlite3
import pandas as pd
import re
import recognizers_suite
from recognizers_suite import Culture, ModelResult
from tqdm import tqdm
import json
pd.options.mode.chained_assignment = None  # default='warn'
culture = Culture.English
## FIXME:sql_tool.py is not a perfect tool to process the table
## It still encounters some problems during the process
## Feel free to raise a pull request if you have a better solution!
## Acknowledgement: We build our sql_tool based on https://github.com/xlang-ai/Binder

def num_tokens_from_string(string, encoding_name = 'cl100k_base'):
    """Returns the number of tokens in a text string."""
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def generate_prompt_with_new_table(case, new_table):
    try:
        instruction = case['prompt']
    except:
        instruction = case['input']
    query = instruction.split('Table\n')[0]
    context = instruction.split('Table\n')[1]
    context_split = context.split('\n\n')

    table, sent_and_task = context_split[0], '\n\n'.join(context_split[1:])
    table_info = []
    for entry in table.split('\n'):
        if 'page title' in entry.lower() or 'caption' in entry.lower() or 'paper title' in entry.lower() or 'section title' in entry.lower():
            table_info.append(entry)
        else:
            break
    table_info = '\n'.join(table_info) + '\n'


    prompt = query + 'Table\n' + table_info + '\n'.join(new_table) + '\n\n' + sent_and_task
    return prompt

def trunacate_input(case, max_tokens = 3200):
    try:
        prompt = case['prompt']
    except:
        prompt = case['input']
    if num_tokens_from_string(prompt) < max_tokens:
        return case
    
    gap = num_tokens_from_string(prompt) - max_tokens
    table = get_table_content_from_instruction(case)
    table = [' | '.join(row) for row in table]
    while gap > 0:
        row = table.pop()
        row_tokens = num_tokens_from_string(row)
        gap -= row_tokens
    
    prompt = generate_prompt_with_new_table(case, table)
    case['prompt'] = prompt
    return case
    

def str_normalize(user_input, recognition_types=None):

    """A string normalizer which recognize and normalize value based on recognizers_suite"""
    user_input = str(user_input)
    user_input = user_input.replace("\\n", "; ")

    def replace_by_idx_pairs(orig_str, strs_to_replace, idx_pairs):
        assert len(strs_to_replace) == len(idx_pairs)
        last_end = 0
        to_concat = []
        for idx_pair, str_to_replace in zip(idx_pairs, strs_to_replace):
            to_concat.append(orig_str[last_end:idx_pair[0]])
            to_concat.append(str_to_replace)
            last_end = idx_pair[1]
        to_concat.append(orig_str[last_end:])
        return ''.join(to_concat)

    if recognition_types is None:
        recognition_types = ["datetime",
                             "number",
                             # "ordinal",
                             # "percentage",
                             # "age",
                             # "currency",
                             # "dimension",
                             # "temperature",
                             ]

    for recognition_type in recognition_types:
        if re.match("\d+/\d+", user_input):
            # avoid calculating str as 1991/92
            continue
        recognized_list = getattr(recognizers_suite, "recognize_{}".format(recognition_type))(user_input,
                                                                                              culture)  # may match multiple parts
        strs_to_replace = []
        idx_pairs = []
        for recognized in recognized_list:
            if not recognition_type == 'datetime':
                recognized_value = recognized.resolution['value']
                if str(recognized_value).startswith("P"):
                    # if the datetime is a period:
                    continue
                else:
                    strs_to_replace.append(recognized_value)
                    idx_pairs.append((recognized.start, recognized.end + 1))
            else:
                if recognized.resolution:  # in some cases, this variable could be none.
                    if len(recognized.resolution['values']) == 1:
                        strs_to_replace.append(
                            recognized.resolution['values'][0]['timex'])  # We use timex as normalization
                        idx_pairs.append((recognized.start, recognized.end + 1))

        if len(strs_to_replace) > 0:
            user_input = replace_by_idx_pairs(user_input, strs_to_replace, idx_pairs)

    if re.match("(.*)-(.*)-(.*) 00:00:00", user_input):
        user_input = user_input[:-len("00:00:00") - 1]
        # '2008-04-13 00:00:00' -> '2008-04-13'
    return user_input


def convert_df_type(df: pd.DataFrame, lower_case=False):
    """
    A simple converter of dataframe data type from string to int/float/datetime.
    """

    def get_table_content_in_column(table):
        if isinstance(table, pd.DataFrame):
            header = table.columns.tolist()
            rows = table.values.tolist()
        else:
            # Standard table dict format
            header, rows = table['header'], table['rows']
        all_col_values = []
        for i in range(len(header)):
            one_col_values = []
            for _row in rows:
                one_col_values.append(_row[i])
            all_col_values.append(one_col_values)
        return all_col_values


    # Rename duplicate columns
    new_columns = []
    for idx, header in enumerate(df.columns):
        if header in new_columns:
            new_header, suffix = header, 2
            while new_header in new_columns:
                new_header = header + '_' + str(suffix)
                suffix += 1
            new_columns.append(new_header)
        else:
            new_columns.append(header)
    df.columns = new_columns

    # Recognize null values like "-"
    null_tokens = ['', '-', '/']
    for header in df.columns:
        df[header] = df[header].map(lambda x: str(None) if x in null_tokens else x)

    # Convert the null values in digit column to "NaN"
    all_col_values = get_table_content_in_column(df)
    for col_i, one_col_values in enumerate(all_col_values):
        all_number_flag = True
        for row_i, cell_value in enumerate(one_col_values):
            try:
                float(cell_value)
            except Exception as e:
                if not cell_value in [str(None), str(None).lower()]:
                    # None or none
                    all_number_flag = False
        if all_number_flag:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: "NaN" if x in [str(None), str(None).lower()] else x)

    # Normalize cell values.
    for header in df.columns:
        df[header] = df[header].map(lambda x: str_normalize(x))

    # Strip the mis-added "01-01 00:00:00"
    all_col_values = get_table_content_in_column(df)
    for col_i, one_col_values in enumerate(all_col_values):
        all_with_00_00_00 = True
        all_with_01_00_00_00 = True
        all_with_01_01_00_00_00 = True
        for row_i, cell_value in enumerate(one_col_values):
            if not str(cell_value).endswith(" 00:00:00"):
                all_with_00_00_00 = False
            if not str(cell_value).endswith("-01 00:00:00"):
                all_with_01_00_00_00 = False
            if not str(cell_value).endswith("-01-01 00:00:00"):
                all_with_01_01_00_00_00 = False
        if all_with_01_01_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[:-len("-01-01 00:00:00")])
            continue

        if all_with_01_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[:-len("-01 00:00:00")])
            continue

        if all_with_00_00_00:
            _header = df.columns[col_i]
            df[_header] = df[_header].map(lambda x: x[:-len(" 00:00:00")])
            continue


    # Recognize header type
    for header in df.columns:

        float_able = False
        int_able = False
        datetime_able = False

        # Recognize int & float type
        try:
            df[header].astype("float")
            float_able = True
        except:
            pass

        if float_able:
            try:
                if all(df[header].astype("float") == df[header].astype(int)):
                    int_able = True
            except:
                pass

        if float_able:
            if int_able:
                df[header] = df[header].astype(int)
            else:
                df[header] = df[header].astype(float)

        # Recognize datetime type
        try:
            df[header].astype("datetime64")
            datetime_able = True
        except:
            pass

        if datetime_able:
            df[header] = df[header].astype("datetime64")

    return df


def process_table_datatype(data):
    for case in data:
        try:
            table = get_table_content_from_instruction(case)
            df = pd.DataFrame(table[1:], columns=table[0])
            df = convert_df_type(df)
            case['df'] = df
        except:
            pass

    return data

def process_table_datatype_transpose(data):
    for case in data:
        try:
            table = get_table_content_from_instruction(case)
            table = list(zip(*table))
            df = pd.DataFrame(table[1:], columns=table[0])
            case['transpose_df'] = df
        except:
            pass
    return data

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def get_table_content_from_instruction(case):
    try:
        instruction = case['input']
    except:
        instruction = case['prompt']
    instruction = instruction.split('Table\n')[1]
    table = instruction.split('\n\n')[0]
    table = table.split('\n')
    while 'page title' in table[0].lower() or 'caption' in table[0].lower() or 'paper title' in table[0].lower() or 'section title' in table[0].lower():
        table = table[1:]
    table = [row.split(' | ') for row in table]
    table = [[cell.strip() for cell in row] for row in table]
    return table

def fill_table_frame(table):
    table.columns = ['col ' + str(idx) if col == '' else col for idx, col in enumerate(table.columns)]
    return table

def get_sql_from_instruction(case):
    try:
        output = case['output']
    except:
        output = case['generated_text']
    sql_list = re.findall(r'```sql\n(.*?)\n```', output, re.DOTALL)

    sql_return_list = []
    for sql in sql_list:
        sql = re.sub(r'FROM \w+', 'FROM w', sql)
        sql = sql.replace('\n', ' ')
        sql_return_list.append(sql)
    return sql_return_list

def fix_sql(table, sql):
    columns = table.columns
    for column in columns:
        if column.strip() == '':
            continue
        def new_column(match, column = column):
            return match.group(1) + '`' + column + '`' + match.group(2)
        pattern =  r'(\s|\(){}(\s|,|\.|\))'.format(re.escape(column))
        sql = re.sub(pattern,  new_column, sql)

        underline_column = '_'.join(column.split(' '))
        pattern =  r'(\s|\(){}(\s|,|\.|\))'.format(re.escape(underline_column))
        sql = re.sub(pattern, new_column, sql)

        merge_column = ''.join(column.split(' '))
        pattern =  r'(\s|\(){}(\s|,|\.|\))'.format(re.escape(merge_column))
        sql = re.sub(pattern,  new_column, sql)
    sql = sql.replace('CHARINDEX', 'instr')
    return sql

def run_sql(sql, table):
    sqlite_conn = sqlite3.connect(':memory:')
    sqlite_conn.row_factory = dict_factory
    table.to_sql('w', sqlite_conn, index=False)
    cursor = sqlite_conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    return result

def execute_sql(case):
    table_dataframe = case['df']
    sql_list = get_sql_from_instruction(case)
    result_list = []
    for sql in sql_list:    
        try:
            table_dataframe = case['df']
            table_dataframe = fill_table_frame(table_dataframe)
            sql = fix_sql(table_dataframe, sql)
            result = run_sql(sql, table_dataframe)
        except:
            table_dataframe = case['transpose_df']
            table_dataframe = fill_table_frame(table_dataframe)
            sql = fix_sql(table_dataframe, sql)
            result = run_sql(sql, table_dataframe)
        result_list.append(result)

    return result_list, sql


def get_sql_result(table):
    header = list(table[0].keys())
    table = [list(row.values()) for row in table]
    table = [header] + table
    table = [[str(cell) for cell in row] for row in table]
    table = '\n'.join([' | '.join(row) for row in table])
    return table

def get_new_instruction_with_sql(case, result_list, sql_list):
    instruction = case['prompt']
    output = case['generated_text']
    output = output.split('```sql\n')
    instruction = instruction + output[0]
    for result, sql in zip(result_list, sql_list):
        if result != []:
            sql_result = get_sql_result(result)
            instruction += f'```sql\n{sql}\n```Execution Result:\n```\n{sql_result}\n```\n'
        else:
            instruction += f'```sql\n{sql}\n```Execution Result:\n\n'
    instruction += '\n3.Step-by-step Answer prediction'

    return instruction

def generate_sql_prompt(case):
    result, sql = execute_sql(case)
    prompt = get_new_instruction_with_sql(case, result, sql)
    return prompt

if __name__ == '__main__':
    path = '../data/outputs/ProTrix/feverous.json'
    data = json.load(open(path))
    data = process_table_datatype(data[:3])
    case = data[0]
    generate_sql_prompt(case)

    process_table_datatype_transpose(data[:3])

    print(trunacate_input(data[2], max_tokens = 500))
