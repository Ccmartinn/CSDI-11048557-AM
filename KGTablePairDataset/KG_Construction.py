from rdflib import Graph, Namespace, Literal, RDF, URIRef
import pandas as pd
import json
import os
import warnings
import requests
from bs4 import BeautifulSoup
import requests
import random
import pandas as pd
from rdflib import Graph, Namespace, URIRef, Literal
import rdflib
import networkx as nx
from pyvis.network import Network

def delete_data_from_table(file_path,doc_num,page_num,delete_proportion):
    
    """
    Divide the dataset into two parts: KG.tsv and Table.tsv

    Args:
        file_path (str): dataset file path
        doc_num (int): ducument number
        page_num (int): page number
        delete_proportion (float): the proportion of KG part

    Returns:
        list: list of row index of KG
    """
    
    df = pd.read_csv(file_path, sep='\t')

    # calculate the number of delete rows
    num_rows_to_delete = int(delete_proportion * len(df))

    # delete 20% of rows randomly
    df_deleted_rows = df.sample(n=num_rows_to_delete, random_state=1)
    delete_row_index = df_deleted_rows.index
    df_remaining = df.drop(delete_row_index)

    # save as KG data
    deleted_rows_output_path = f'Modified_WTQ/{doc_num}-csv/{page_num}_kg.tsv'
    df_deleted_rows.reset_index().to_csv(deleted_rows_output_path, sep='\t', index=False)

    # save as Table data
    remaining_rows_output_path = f'Modified_WTQ/{doc_num}-csv/{page_num}_table.tsv'
    df_remaining.reset_index().to_csv(remaining_rows_output_path, sep='\t', index=False)
    
    return delete_row_index


def generate_ttl(tsv_file_path,json_file_path,output_ttl):
    """
    Generates a Turtle (TTL) file from a TSV file and a JSON metadata file.

    Args:
        tsv_file_path (str): Path to the TSV file containing table data.
        json_file_path (str): Path to the JSON file with metadata (e.g., table title).
        output_ttl (str): Path to the output TTL file.

    Returns:
        str: The main entity name used as the identifier in the TTL file.
    """
    df = pd.read_csv(tsv_file_path, sep='\t')
    # print(df)

    # add prefix
    g = Graph()
    WTQ = Namespace("http://wikitablequestion.org/")
    WD = Namespace("http://www.wikidata.org/entity/")
    WDT = Namespace("http://www.wikidata.org/prop/direct/")
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    
    g.bind("wtq", WTQ)
    g.bind("wd", WD)
    g.bind("wdt", WDT)
    g.bind("rdfs", RDFS)

    # json_file_path = 'WTQ/'+str(csv_num) + '-csv/' + str(tab_num) +'.json'
    # json_file_path = 'WTQ/203-csv/52.json'

    with open(json_file_path, 'r', encoding='utf-8') as file:
        pages_json = json.load(file)
    title = pages_json.get('title', 'Title not found')
    main_entity_name = title.replace(" ","_")
        
    # define head of table
    table_entity = WTQ[main_entity_name]
    g.add((table_entity, RDF.type, WTQ.Entity))

    # Iterate over the dataframe and create triples
    for index, row in df.iterrows():
        row_entity = WTQ[f"{main_entity_name}_row_{row['index']}"]
        g.add((table_entity, WTQ.hasRow, row_entity))
        
        for col in df.columns:
            col_name = col.replace('(', '_').replace(')', '_') \
                .replace(' ', '_').replace('/', '_').replace('\\', '_') 
            property_uri = WTQ[f"has{col_name}"]
            value = Literal(str(row[col]))  # Ensure all values are literals
            g.add((row_entity, property_uri, value))
            
    # Serialize the graph in the desired format
    rdf_data = g.serialize(format='turtle')
    # print(rdf_data)

    g.parse(data=rdf_data, format="turtle")

    #Save the graph
    g.serialize(destination = output_ttl)

    return main_entity_name


def extract_wikipedia_links(html_file):
    
    """
    Extracts Wikipedia links from an HTML file containing a table.

    Args:
        html_file (str): Path to the HTML file.

    Returns:
        dict: A dictionary with cell positions as keys (row, col) and lists of (URL, text) tuples as values.
    """
    
    with open(html_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    rows = soup.find_all('tr')

    cell_links = {}
    for row_index, row in enumerate(rows[1:]):  
        cells = row.find_all(['td', 'th'])
        for col_index, cell in enumerate(cells):
            links = cell.find_all('a', href=True)
            hrefs = []
            for a in links:
                href = a['href']
                text = a.get_text(strip=True)
                if href.startswith("//en.wikipedia.org/wiki/") and text:
                    hrefs.append(("https:" + href, text))
            if hrefs:
                cell_links[(row_index, col_index)] = hrefs

    return cell_links


def get_wikidata_entity_and_label_from_wikipedia_url(wikipedia_url):
    
    """
    Retrieves the Wikidata entity and its label corresponding to a given Wikipedia URL.

    Args:
        wikipedia_url (str): The URL of the Wikipedia article.

    Returns:
        tuple: A tuple containing the Wikidata entity URI (str) and the label (str).
               If no corresponding entity is found, returns (None, None).
    """
    
    article_name = wikipedia_url.split("/")[-1].replace("_", " ")
    
    query = """
    SELECT ?item ?itemLabel WHERE {{
      ?article schema:about ?item ;
               schema:inLanguage "en" ;
               schema:isPartOf <https://en.wikipedia.org/> ;
               schema:name "{}"@en .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT 1
    """.format(article_name)
    
    url = "https://query.wikidata.org/sparql"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, params={'query': query, 'format': 'json'}, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', {}).get('bindings', [])
        if results:
            return results[0]['item']['value'], results[0]['itemLabel']['value']
    return None, None

def get_valid_entity(cell_links):
    """
    Retrieves valid Wikidata entities and labels for Wikipedia URLs in each cell.

    Args:
        cell_links (dict): A dictionary with cell identifiers as keys and lists of (URL, text) tuples as values.

    Returns:
        dict: A dictionary with cell identifiers as keys and lists of (entity URI, label) tuples for valid entities.
    """
    wikidata_entities_and_labels_per_cell = {}
    for cell, links in cell_links.items():
        entities_and_labels = []
        for wikipedia_url, text in links:
            wikidata_entity, wikidata_label = get_wikidata_entity_and_label_from_wikipedia_url(wikipedia_url)
            if wikidata_entity and wikidata_label:
                entities_and_labels.append((wikidata_entity, wikidata_label))
        if entities_and_labels:
            wikidata_entities_and_labels_per_cell[cell] = entities_and_labels
    return wikidata_entities_and_labels_per_cell



def query_wikidata_entity_relationships(entity_id):
    """
    Queries Wikidata for relationships of a given entity and returns a random selection of up to three.

    Args:
        entity_id (str): The Wikidata entity ID (e.g., "Q42").

    Returns:
        list: A list of up to three tuples, each containing a property label (str) and value label (str).
              Returns None if the query fails.
    """
    query = f"""
    SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
      wd:{entity_id} ?property ?value .
      FILTER (STRSTARTS(STR(?value), "http://www.wikidata.org/entity/Q")) .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    LIMIT 10
    """
    
    url = "https://query.wikidata.org/sparql"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(url, params={'query': query, 'format': 'json'}, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', {}).get('bindings', [])
        relationships = []
        for result in results:
            property_label = result['propertyLabel']['value']
            value_label = result['valueLabel']['value']
            relationships.append((property_label, value_label))
        
        if len(relationships) > 3:
            relationships = random.sample(relationships, 3)
        
        return relationships
    else:
        return None

def get_entity_relationships(entity_url):
    """
    Retrieves relationships for a given Wikidata entity from its URL.

    Args:
        entity_url (str): The URL of the Wikidata entity.

    Returns:
        list: A list of up to three tuples, each containing a property label (str) and value label (str).
              Returns None if the query fails.
    """ 
    entity_id = entity_url.split('/')[-1]
    relationships = query_wikidata_entity_relationships(entity_id)
    return relationships

def process_table_with_relationships(wikidata_entities_and_labels_per_cell):
    """
    Processes Wikidata entities in table cells to retrieve their relationships.

    Args:
        wikidata_entities_and_labels_per_cell (dict): A dictionary with cell identifiers as keys and lists of 
                                                      (entity URL, entity label) tuples as values.

    Returns:
        dict: A dictionary where keys are cell identifiers and values are lists of dictionaries.
    """
    results = {}
    for cell, entities in wikidata_entities_and_labels_per_cell.items():
        for entity_url, entity_label in entities:
            relationships = get_entity_relationships(entity_url)
            if relationships:
                if cell not in results:
                    results[cell] = []
                results[cell].append({
                    'entity': [(entity_url,entity_label)],
                    'relationships': relationships
                })
    return results

def generate_cleaned_ttl(output_cleaned_ttl, tsv_file, output_ttl, results, delete_row_index, main_entity_name):
    df = pd.read_csv(tsv_file, sep='\t')
    """
    Updates and generates a cleaned TTL file with additional RDF data from provided results.

    Args:
        output_cleaned_ttl (str): Path to save the cleaned TTL file.
        tsv_file (str): Path to the TSV file containing table data.
        output_ttl (str): Path to the existing TTL file to be updated.
        results (dict): A dictionary containing entity relationships for specific table cells.
        delete_row_index (pd.Series): A series of row indices to be considered for deletion or update.
        main_entity_name (str): The main entity name used as the identifier in the TTL file.

    Returns:
        None
    """
    g = Graph()
    g.parse(output_ttl, format="ttl")
    
    WTQ = Namespace("http://wikitablequestion.org/")
    WD = Namespace("http://www.wikidata.org/entity/")
    WDT = Namespace("http://www.wikidata.org/prop/direct/")
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    
    g.bind("wtq", WTQ)
    g.bind("wd", WD)
    g.bind("wdt", WDT)
    g.bind("rdfs", RDFS)

    for (row_index, col_index), entities_data in results.items():
        if row_index in delete_row_index.values:
            row_entity = WTQ[f"{main_entity_name}_row_{row_index}"]
            col_name = df.columns[col_index + 1].replace('(', '_').replace(')', '_') \
                .replace(' ', '_').replace('/', '_').replace('\\', '_')
                
            property_uri = WTQ[f"has{col_name}"]
            for entity_data in entities_data:
                for entity, label in entity_data['entity']:
                    entity_uri = URIRef(entity)
                    g.add((row_entity, property_uri, entity_uri))
                    g.add((entity_uri, RDFS.label, Literal(label)))
                
                # Adding relationships and their labels
                for relationship in entity_data['relationships']:
                    rel_property, rel_value = relationship
                    rel_property_uri = URIRef(rel_property)
                    rel_value_literal = Literal(rel_value)
                    g.add((entity_uri, rel_property_uri, rel_value_literal))

    g.serialize(destination=output_cleaned_ttl, format='ttl')

def draw_kg(file_path,output_kg_graph):
    """
    Generates and visualizes a knowledge graph from an RDF Turtle file.

    Args:
        file_path (str): Path to the input Turtle (TTL) file.
        output_kg_graph (str): Path to save the generated knowledge graph visualization (HTML format).

    Returns:
        None
    """
    # Load the RDF graph from the Turtle file using rdflib
    rdf_graph = rdflib.Graph()
    rdf_graph.parse(file_path, format="ttl")

    # Define namespace prefix replacements
    NAMESPACE_REPLACEMENTS = {
        "http://wikitablequestion.org/": "wtq:",
        "http://www.wikidata.org/entity/": "wd:",
        "http://www.wikidata.org/prop/direct/": "wdt:",
        "http://www.w3.org/2000/01/rdf-schema#":"rdfs:"
    }

    def replace_namespace(uri):
        for namespace, prefix in NAMESPACE_REPLACEMENTS.items():
            if uri.startswith(namespace):
                return uri.replace(namespace, prefix)
        return uri
    
    literal_counter = 0
    # Create a new NetworkX graph to hold the final structure
    final_graph = nx.Graph()

    # Add all URI nodes and their edges
    for subj, pred, obj in rdf_graph:
        subj_str = replace_namespace(str(subj))
        pred_str = replace_namespace(str(pred))
        if isinstance(obj, rdflib.Literal):
            obj_str = f"literal_{literal_counter}_{str(obj)}"
            literal_counter += 1
        else:
            obj_str = replace_namespace(str(obj))
        
        final_graph.add_node(subj_str, label=subj_str, type="uri")
        
        if not isinstance(obj, rdflib.Literal):
            final_graph.add_node(obj_str, label=obj_str, type="uri")
            final_graph.add_edge(subj_str, obj_str, label=pred_str)

    # Add each literal node and connect it to only one URI node
    literal_counter = 0
    for subj, pred, obj in rdf_graph:
        if isinstance(obj, rdflib.Literal):
            subj_str = replace_namespace(str(subj))
            obj_str = f"literal_{literal_counter}_{str(obj)}"
            literal_counter += 1
            
            if not final_graph.has_node(obj_str):
                final_graph.add_node(obj_str, label=str(obj), type="literal")
                final_graph.add_edge(subj_str, obj_str, label=replace_namespace(str(pred)))

    # Define visualization styles
    VIS_STYLE = {
        "literal": {
            "color": "gray",
            "size": 30,
        },
        "uri": {
            "color": "green",
            "size": 40,
        }
    }

    # Create a pyvis network
    pyvis_graph = Network(notebook=True)

    # Add nodes and edges with styles
    for node, data in final_graph.nodes(data=True):
        node_type = data.get("type", "uri")
        style = VIS_STYLE.get(node_type, VIS_STYLE["uri"])
        pyvis_graph.add_node(node, label=data["label"], **style)

    for source, target, data in final_graph.edges(data=True):
        pyvis_graph.add_edge(source, target, title=data["label"])

    pyvis_graph.force_atlas_2based()
    pyvis_graph.show(output_kg_graph)

# Suppress all warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    delete_proportion = 0.4
    doc_num = 203
    
    if os.path.exists(f'WTQ/{doc_num}-csv'):
        # List all files in the directory
        all_files = os.listdir(f'WTQ/{doc_num}-csv')

        tsv_files = [int(os.path.splitext(file)[0]) for file in all_files if file.endswith('.tsv')]
    else:
        tsv_files = []
   
    # tsv_files = [105]
    for page_num in [90]:
        
        print(f"-----------------------------------------------")

        print(f"Table {doc_num}-{page_num} Processing")
        
        org_tsv_file_path = f'WTQ/{doc_num}-csv/{page_num}.tsv'
        delete_row_index = delete_data_from_table(org_tsv_file_path,doc_num,page_num,delete_proportion)
        
        tsv_file_path = f'Modified_WTQ/{doc_num}-csv/{page_num}_kg.tsv'
        json_file_path = f'WTQ/{doc_num}-csv/{page_num}.json'
        html_file = f'WTQ/{doc_num}-csv/{page_num}.html'
        
        output_ttl = f'Modified_WTQ/kg-{doc_num}-csv/{page_num}.ttl'
        output_cleaned_ttl = f'Modified_WTQ/kg-{doc_num}-csv/{page_num}_cleaned.ttl'
        
        main_entity_name = generate_ttl(tsv_file_path,json_file_path,output_ttl)
        cell_links = extract_wikipedia_links(html_file)
        wikidata_entities_and_labels_per_cell = get_valid_entity(cell_links)
        
        results = process_table_with_relationships(wikidata_entities_and_labels_per_cell)
        
        generate_cleaned_ttl(output_cleaned_ttl, tsv_file_path, output_ttl, results, delete_row_index, main_entity_name)
        # generate_cleaned_ttl(output_cleaned_ttl,tsv_file_path,output_ttl,wikidata_entities_and_labels_per_cell,delete_row_index,main_entity_name)
        
        output_kg_graph =  f'Modified_WTQ/kg-{doc_num}-csv/{page_num}.html'
        draw_kg(output_cleaned_ttl,output_kg_graph)
        
        print(f"Table {doc_num}-{page_num} Processing Complete")
        
        
