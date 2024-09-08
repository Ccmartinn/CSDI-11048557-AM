import os
import csv
import json
import shutil
import pandas as pd
from openai import OpenAI
from collections import defaultdict
from config import api_key  # use your own chatgpt api key
from rdflib import Graph, Namespace, URIRef, Literal

def Generate_table_triples_tree(json_file, tsv_file):
    # load document
    with open(json_file, 'r') as f:
        json_data = json.load(f)
        title = json_data["title"]
    
    tsv_file_path = tsv_file
    df = pd.read_csv(tsv_file_path, sep='\t')

    # Create triples dict
    tree = defaultdict(list)

    for _, row in df.iterrows():
        index = row['index']  
        for col in df.columns:
            if col != 'index':
                triple = (f"{title} - Index {index}", col, row[col])
                tree[f"{title} - Index {index}"].append(triple)
    
    triples_tree = dict(tree)
    
    return triples_tree,title

def generate_qa_set():
    training_df = pd.read_csv('training.tsv', sep='\t')
    question_df = pd.read_excel('question.xlsx', sheet_name='Sheet1')
    question_subset = question_df[['question_num', 'types(1-table, 2-kg, 3-table+kg)']]
    question_subset['question_num'] = question_subset['question_num'].astype(str)
    question_subset['training_id'] = 'nt-' + question_subset['question_num']
    merged_df = pd.merge(question_subset, training_df, left_on='training_id', right_on='id', how='inner')
    output_df = merged_df[['question_num','types(1-table, 2-kg, 3-table+kg)','utterance','context','targetValue']]
    output_df['csv_file'] = 203
    output_df['context'] = output_df['context'].str.extract(r'csv/203-csv/(\d+)\.csv')
    output_df.rename(columns={'types(1-table, 2-kg, 3-table+kg)': 'type',
                            'context':'page'}, inplace=True)
    return output_df

def generate_natural_language_from_kg_dict(triples_tree,title):
    client = OpenAI(api_key=api_key)

    all_triples = []
    index_mapping = {}
    separator = "\n---\n"  
    for index, triples in triples_tree.items():
        triples_str = '\n'.join([f'({sub}, {pred}, {obj})' for sub, pred, obj in triples])
        all_triples.append(triples_str)
        index_mapping[len(all_triples) - 1] = index 

    all_triples_str = separator.join(all_triples)

    response = client.chat.completions.create(
    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": (
                "Convert the following triples into natural language descriptions, "
                "keeping each description separate and formatted similarly:\n" + 
                "Please format the output as:\n" 
                + title + "Index <number> : <text>\n" +
                "Separate each line with a new line." +
                "\n\n" + all_triples_str
            )}
        ],
        model="gpt-4-turbo",
        max_tokens=1000,  
        temperature=0.7,
    )
    response_text = response.choices[0].message.content.strip()
    lines = response_text.split('\n')
    result_dict = {}
    for line in lines:
        if " : " in line:
            key, value = line.split(" : ", 1)
            result_dict[key.strip()] = value.strip()

    return result_dict

def Generate_KG_triples_tree(ttl_file):
    g = Graph()
    g.parse(ttl_file, format="ttl")

    wtq = Namespace("http://wikitablequestion.org/")
    wd = Namespace("http://www.wikidata.org/entity/")
    rdfs = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    wdt = Namespace("http://www.wikidata.org/prop/direct/")

    wtq_dict = {}

    for s, p, o in g:
        if isinstance(s, URIRef) and s.startswith(wtq):
            entity_key = str(s).replace(str(wtq), "")
            
            if entity_key not in wtq_dict:
                wtq_dict[entity_key] = []

            # remove prefix wtq、rdfs、wdt、wd
            if str(p).startswith(str(wtq)):
                predicate = str(p).replace(str(wtq), "")
            elif str(p).startswith(str(rdfs)):
                predicate = str(p).replace(str(rdfs), "")
            elif str(p).startswith(str(wdt)):
                predicate = str(p).replace(str(wdt), "")
            elif str(p).startswith(str(wd)):
                predicate = str(p).replace(str(wd), "")
            else:
                predicate = str(p)

            # Handling the case where the object has a prefix
            if isinstance(o, URIRef):
                if o.startswith(wtq):
                    # remove WTQ
                    o_value = str(o).replace(str(wtq), "")
                elif o.startswith(wd):
                    # Remove the wd prefix and append all triples with wd as the subject
                    o_value_key = str(o).replace(str(wd), "")
                    nested_triples = []
                    for wd_s, wd_p, wd_o in g.triples((o, None, None)):
                        nested_predicate = str(wd_p).replace(str(wtq), "").replace(str(rdfs), "").replace(str(wdt), "").replace(str(wd), "")
                        nested_object = str(wd_o).replace(str(wtq), "").replace(str(rdfs), "").replace(str(wdt), "").replace(str(wd), "")
                        nested_triples.append({nested_predicate: nested_object})
                    
                    o_value = {
                        o_value_key: nested_triples
                    }
                    o_value = json.dumps(o_value, ensure_ascii=False)
                elif o.startswith(wdt):
                    # remove WDT
                    o_value = str(o).replace(str(wdt), "")
                else:
                    o_value = str(o)
            else:
                o_value = str(o)

            # If predicate and object are the same, skip
            if (entity_key, predicate, o_value) not in wtq_dict[entity_key]:
                wtq_dict[entity_key].append((entity_key, predicate, o_value))
    
    keys_list = list(wtq_dict.keys())

    shortest_key = min(keys_list, key=len)

    if shortest_key in wtq_dict:
        del wtq_dict[shortest_key]

    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(wtq_dict, f, ensure_ascii=False, indent=4)
        
    return wtq_dict

def save_knowledge_to_file(knowledge_dict, filename):
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(knowledge_dict, json_file, ensure_ascii=False, indent=4)
    print(f"Knowledge saved to {filename}")


def copy_file(source_dir,target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        if filename.endswith('_table.tsv'):
        # if filename.endswith('_cleaned.ttl'):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            
            shutil.copy2(source_file, target_file)
            print(f"File {filename} has been copied to {target_dir}")

def load_natural_language(num):
    current_dir = os.getcwd() 
    file_path = os.path.join(current_dir, 'NL', f'knowledge_{num}_gpt4_turbo.json')

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    else:
        print(f"File {file_path} Not Exist")
        
def load_KG(num):
    current_dir = os.getcwd() 
    file_path = os.path.join(current_dir, 'NL', f'KG_{num}_gpt4_turbo.json')

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    else:
        print(f"File {file_path} Not Exist")
    
def load_Table(num):
    current_dir = os.getcwd() 
    file_path = os.path.join(current_dir, 'NL', f'Table_{num}_gpt4_turbo.json')

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    else:
        print(f"File {file_path} Not Exist")
        


def copy_file(source_dir,target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for filename in os.listdir(source_dir):
        if filename.endswith('_table.tsv'):
        # if filename.endswith('_cleaned.ttl'):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            
            shutil.copy2(source_file, target_file)
            print(f"File {filename} has been copied to {target_dir}")

def load_natural_language(num):
    current_dir = os.getcwd() 
    file_path = os.path.join(current_dir, 'NL', f'knowledge_{num}_gpt4_turbo.json')

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    else:
        print(f"File {file_path} Not Exist")
        
def load_KG(num):
    current_dir = os.getcwd() 
    file_path = os.path.join(current_dir, 'NL', f'KG_{num}_gpt4_turbo.json')

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    else:
        print(f"File {file_path} Not Exist")
    
def load_Table(num):
    current_dir = os.getcwd() 
    file_path = os.path.join(current_dir, 'NL', f'Table_{num}_gpt4_turbo.json')

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    else:
        print(f"File {file_path} Not Exist")

def generate_answers(knowledge_dict, questions):
    client = OpenAI(api_key=api_key)
    knowledge = "\n".join([f"{title}: {content}" for title, content in knowledge_dict.items()])
    question_block = "\n".join([f"Q{index+1}: {question}" for index, question in enumerate(questions)])
    prompt = (
        f"Given the following knowledge:\n{knowledge}\n\n"
        f"Answer the following questions:\n{question_block}\n\n"
        "Answers:"
    )
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a knowledgeable assistant and your answers should be concise."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-4-turbo",
        max_tokens=1000, 
        n=1,
        stop=None,
        temperature=0.7,
    )
    answers = response.choices[0].message.content.strip().split('\n')
    return answers

def generate_answers_for_KG_table_pair(knowledge_dict,question_df,num):
    questions = []
    for index, row in question_df.iterrows():
        if row['page'] == num:
            questions.append(row['utterance'])

    answers = generate_answers(knowledge_dict, questions)
    print(answers)
    
    # save answer as csv file with utf-8 encoding
    filename = f'./Answer/{num}_answer.csv'
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer'])  
        for index, answer in enumerate(answers):
            writer.writerow([f"Q{index+1}", answer])
            
    print(f'Answer of Pair {num} have been saved Successfully!')
    
    return answers


def insert_answers(question_df, num):
    matching_rows = question_df[question_df['page'] == num]
    answer_list = pd.read_csv(f'./Answer/{num}_answer.csv')
    if len(matching_rows) != len(answer_list):
        raise ValueError("The number of answers does not match the number of matching rows in question_df.")
    question_df.loc[question_df['page'] == num, 'Answer'] = answer_list['Answer'].values
    return question_df
    
def load_answer(num):
    answer_list = pd.read_csv(f'./Answer/{num}_answer.csv')
    return answer_list

def insert_answers(question_df,answer_list, num):
    matching_rows = question_df[question_df['page'] == num]
    answer_list = pd.read_csv(f'./Answer/{num}_answer.csv')
    if len(matching_rows) != len(answer_list):
        raise ValueError("The number of answers does not match the number of matching rows in question_df.")
    question_df.loc[question_df['page'] == num, 'Answer'] = answer_list['Answer'].values
    return question_df

def save_answer(question_df):
    filename = f'./Answer/all_answer.csv'
    question_df.to_csv(filename, index=False, encoding='utf-8')


if __name__ == "__main__":
    # generate sorce data num list
    json_files = [int(os.path.splitext(f)[0]) for f in os.listdir('./SourceData') if f.endswith('.json')]
    json_files_sorted = sorted(json_files)
    # load question set
    question_df = pd.read_csv('question_set.csv')
    
    for num in json_files_sorted:
        # generate natural language for table 
        triples_tree,title = Generate_table_triples_tree(f'./SourceData/{num}.json', f'./SourceData/{num}_table.tsv')
        table_nl = generate_natural_language_from_kg_dict(triples_tree,title)
        save_knowledge_to_file(table_nl, f'./NL/Table_{num}_gpt4_turbo.json')
        print(f'Table {num} processing completed')
        # table_nl = load_Table(num)
        
        # # generate natural language for KG 
        wtq_dict = Generate_KG_triples_tree(f'./SourceData/{num}_cleaned.ttl')
        kg_nl = generate_natural_language_from_kg_dict(wtq_dict,title)
        save_knowledge_to_file(kg_nl, f'./NL/KG_{num}_gpt4_turbo.json')
        print(f'KG {num} processing completed')
        # kg_nl = load_KG(num)
        
        # merge them together 
        knowledge_nl = {**table_nl, **kg_nl}
        save_knowledge_to_file(knowledge_nl, f'./NL/knowledge_{num}_gpt4_turbo.json')
        print(f'Knowledge {num} processing completed')
        
        json_files = [int(os.path.splitext(f)[0]) for f in os.listdir('./SourceData') if f.endswith('.json')]

    csv_files = [os.path.splitext(f)[0] for f in os.listdir('./Answer') if f.endswith('.csv')]

    for num in json_files_sorted:
        knowledge_nl = load_natural_language(num)
        answer_list = generate_answers_for_KG_table_pair(knowledge_nl,question_df,num)
        print(f'{num} have been processed')
        
        for file in [f'{num}_answer']:
            df = pd.read_csv(f'./Answer/{file}.csv')
            df_cleaned = df.dropna(subset=['Answer'])
            df_cleaned['Question'] = ['Q' + str(i+1) for i in range(len(df_cleaned))]
            df_cleaned.to_csv(f'./Answer/{file}.csv', index=False)
    
    
    for num in json_files_sorted:
        # print(num)
        answer_list = load_answer(num)
        question_df = insert_answers(question_df,answer_list, num)

        # print(question_df[question_df['page'] == num]['Answer'])   
        
    save_answer(question_df)

### bk version: generate natural language by each row
# def generate_natural_language_from_kg_dict(triples_tree):
#     client = OpenAI(api_key= api_key)
    
#     # use the result of first tripes as sample
#     first_index = list(triples_tree.keys())[0]
#     first_triples = triples_tree[first_index]
#     triples_str = '\n'.join([f'({sub}, {pred}, {obj})' for sub, pred, obj in first_triples])

#     first_response = client.chat.completions.create(
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": f"Convert the following triples into a natural language description:\n{triples_str}"}
#         ],
#         model="gpt-4-turbo",
#     )
#     example_description = first_response.choices[0].message.content.strip()

#     responses = {}
#     for index, triples in triples_tree.items():
#         if index == first_index:
#             responses[index] = example_description 
#             continue
        
#         triples_str = '\n'.join([f'({sub}, {pred}, {obj})' for sub, pred, obj in triples])
        
#         response = client.chat.completions.create(
#             model="gpt-4-turbo",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {"role": "user", "content": (
#                     "Convert the following triples into a natural language description using the following format:\n\n"
#                     f"{example_description}\n\n"
#                     "Here are the triples:\n" + triples_str
#                 )}
#             ]
#         )
#         print(f'row {index} finish')
#         responses[index] = response.choices[0].message.content.strip()

#     return responses

