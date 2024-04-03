import gzip
import json
import math
import os
import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer

ps = PorterStemmer()

index_file = 'final_index_stemmed.bin'
catalog_file = 'final_catalog_stemmed.json'

# compress index files to .gz
# with (open(index_file, 'rb') as input_file,
#       gzip.open('final_index.gz', 'wb') as output_file):
#     output_file.writelines(input_file)

# read stoplist.txt as a set
stopwords_path = 'Resources/stoplist.txt'
with open(stopwords_path) as file:
    stopwords = set(file.read().splitlines())


# remove stopwords
def remove_stopwords(text):
    updated_text = ' '.join([word for word in text.lower().split() if word not in stopwords])
    return updated_text


# stem the text
def stem_text(text, ps):
    words = word_tokenize(text)
    updated = [ps.stem(word) for word in words]
    stemmed = ' '.join(updated)
    return stemmed


# parse the queries
query_path = 'Resources/queries.txt'
queries = {}

with open(query_path) as file:
    for line in file:
        dot = line.find('.')
        query_number = int(line[:dot].strip())
        query_text = line[dot + 1:].strip()
        query_text = ' '.join(query_text.split(' '))
        query_text = ''.join(c for c in query_text if c.isalnum() or c.isspace() or c == '-')
        query_text = query_text.replace('-', ' ')
        queries[query_number] = stem_text(remove_stopwords(query_text), ps)


doc_no_pattern = re.compile(r"<DOCNO>(.+?)</DOCNO>")
text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

ap89_num = 1
doc_text_dict = {}

# extract text from document
def parse_file(path):
    print("Parsing ap89 files...", ap89_num)
    with open(path, 'r') as f:
        lines = f.readlines()
    doc_num = ""
    content = ""
    in_text = False
    for line in lines:
        if line.startswith("<DOCNO>"):
            match = doc_no_pattern.search(line)
            if match:
                doc_num = match.group(1).strip()
        elif line.startswith("<TEXT>"):
            in_text = True
            text_match = text_pattern.search(line)
            if text_match:
                content += text_match.group(1)
        elif in_text and "</TEXT>" not in line:
            content += re.sub(r'\b\.(?![a-zA-Z0-9])|\.\.+|[^\w\s.]', ' ', line)
        elif in_text and "</TEXT>" in line:
            in_text = False
        elif line.startswith("</DOC>"):
            content = stem_text(remove_stopwords(content), ps)
            doc_text_dict[doc_num] = content.strip().split()
            doc_num = ""
            content = ""


# go through each file and parse the documents
for filename in os.listdir("Resources/ap89_collection"):
    if filename.lower() != 'readme':
        parse_file(os.path.join("Resources/ap89_collection", filename))
        ap89_num += 1


# get term cf and df json info
with open('Stemmed/term_cf.json', 'r') as json_file:
    term_cf = json.load(json_file)
with open('Stemmed/term_df.json', 'r') as json_file:
    term_df = json.load(json_file)

# get doc id and term id json info
with open('Stemmed/doc_to_id.json', 'r') as json_file:
    doc_ids = json.load(json_file)
with open('Stemmed/id_to_doc.json', 'r') as json_file:
    id_docs = json.load(json_file)
with open('Stemmed/term_to_id.json', 'r') as json_file:
    term_ids = json.load(json_file)
with open('Stemmed/id_to_term.json', 'r') as json_file:
    id_terms = json.load(json_file)

# vocab size (number of unique terms in collection)
vocab_size = len(id_terms)
print("# of unique terms in collection (VOCAB SIZE):", vocab_size)

# total cf (total number of tokens in collection)
total_cf = sum(len(value) for value in doc_text_dict.values())
print("Total number of tokens (TOTAL CF):", total_cf)

# store document lengths and average doc length
try:
    with open('Stemmed/doc_length.json', 'r') as json_file:
        doc_length = json.load(json_file)
except (FileNotFoundError, json.decoder.JSONDecodeError):
    doc_length = {}
    for doc, content in doc_text_dict.items():
        doc_id = doc_ids[doc]
        doc_length[doc_id] = len(content)
    with open('Stemmed/doc_length.json', 'w') as file:
        json.dump(doc_length, file, indent=2)

avg_doc_length = total_cf / len(doc_length)
print('Average document length: ', avg_doc_length)

# load the catalog (catalog is a list of entries)
with open(catalog_file, 'r') as json_file:
    catalog = json.load(json_file)


# tf-idf
def run_tf_idf():
    for query_number, query_text in queries.items():
        output_lines = []
        for doc_id, doc_name in id_docs.items():
            total_score = 0
            for term in query_text.split():
                # get term frequency
                if term not in term_ids:
                    continue
                else:
                    term_id = term_ids[term]
                    catalog_data = next((entry for entry in catalog if entry[0] == term_id), None)
                    with open(index_file, "rb") as file:
                        file.seek(catalog_data[1])
                        bin_data = file.read(catalog_data[2])
                        data = json.loads(bin_data)
                    for term_id, doc_maps in data.items():
                        tf = len(doc_maps.get(str(doc_id), []))
                    df = term_df.get(str(term_id))
                den = 0.5 + 1.5 * (doc_length.get(str(doc_id)) / avg_doc_length)
                okapi = tf / (tf + den) if (tf + den) != 0 else 0
                total_score += okapi * math.log(len(id_docs) / max(df, 1))
            output_lines.append((total_score, doc_name))
            print("tf idf", query_number, doc_id, total_score)

        # add to output file after each query
        output_lines.sort(reverse=True, key=lambda x: x[0])
        with open('Resources/stemmed_query_result_tf_idf.txt', 'a') as file:
            for rank, (score, doc_id) in enumerate(output_lines[:1000], start=1):
                ranked_line = f"{query_number} Q0 {doc_id} {rank} {score} Exp\n"
                file.write(ranked_line)


run_tf_idf()

# okapi bm25
def run_okapi_bm25():
    k1 = 1.2
    k2 = 100
    b = 0.75

    for query_number, query_text in queries.items():
        output_lines = []
        for doc_id, doc_name in id_docs.items():
            total_score = 0
            for term in query_text.split():
                # get term frequency
                if term not in term_ids:
                    continue
                else:
                    term_id = term_ids[term]
                    catalog_data = next((entry for entry in catalog if entry[0] == term_id), None)
                    with open(index_file, "rb") as file:
                        file.seek(catalog_data[1])
                        bin_data = file.read(catalog_data[2])
                        data = json.loads(bin_data)
                    for term_id, doc_maps in data.items():
                        tf = len(doc_maps.get(str(doc_id), []))
                    df = term_df.get(str(term_id))
                first = math.log((len(id_docs) + 0.5) / (df + 0.5))
                second = (tf + k1 * tf) / (tf + k1 * ((1 - b) + b * (doc_length.get(doc_id) / avg_doc_length)))
                third = (tf + k2 * tf) / (tf + k2)
                total_score += first * second * third
            output_lines.append((total_score, doc_name))
            print("bm25", query_number, doc_id, total_score)

        # add to output file after each query
        output_lines.sort(reverse=True, key=lambda x: x[0])
        with open('Resources/stemmed_query_result_okapi_bm.txt', 'a') as file:
            for rank, (score, doc_id) in enumerate(output_lines[:1000], start=1):
                ranked_line = f"{query_number} Q0 {doc_id} {rank} {score} Exp\n"
                file.write(ranked_line)


run_okapi_bm25()

# unigram LM with laplace
def run_laplace():
    for query_number, query_text in queries.items():
        output_lines = []
        for doc_id, doc_name in id_docs.items():
            total_score = 0
            for term in query_text.split():
                # get term frequency
                if term not in term_ids:
                    tf = -1000
                else:
                    term_id = term_ids[term]
                    catalog_data = next((entry for entry in catalog if entry[0] == term_id), None)
                    with open(index_file, "rb") as file:
                        file.seek(catalog_data[1])
                        bin_data = file.read(catalog_data[2])
                        data = json.loads(bin_data)
                    for term_id, doc_maps in data.items():
                        tf = len(doc_maps.get(str(doc_id), []))
                score = (tf + 1) / (doc_length.get(str(doc_id)) + vocab_size)
                if score > 0:
                    total_score += math.log(score)
                else:
                    total_score += -1000
            output_lines.append((total_score, doc_name))
            print("laplace", query_number, doc_id, total_score)

        # add to output file after each query
        output_lines.sort(reverse=True, key=lambda x: x[0])
        valid_lines_written = 0
        with open('Resources/stemmed_query_result_laplace.txt', "a") as file:
            for rank, (score, doc_id) in enumerate(output_lines, start=1):
                if score != 0:
                    ranked_line = f"{query_number} Q0 {doc_id} {rank} {score} Exp\n"
                    file.write(ranked_line)
                    valid_lines_written += 1
                if valid_lines_written == 1000:
                    break


run_laplace()
