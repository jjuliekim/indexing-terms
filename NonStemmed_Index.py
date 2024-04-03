import os
import re

import json

# dict with doc id and tokenized text
doc_text_dict = {}
folder = "Resources/ap89_collection"

# read stoplist.txt as a set
stopwords_path = 'Resources/stoplist.txt'
with open(stopwords_path) as file:
    stopwords = set(file.read().splitlines())


# remove stopwords
def remove_stopwords(text):
    updated_text = ' '.join([word for word in text.lower().split() if word not in stopwords])
    return updated_text


doc_no_pattern = re.compile(r"<DOCNO>(.+?)</DOCNO>")
text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL)

ap89_num = 1


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
            content = remove_stopwords(content)
            doc_text_dict[doc_num] = content.strip().split()
            doc_num = ""
            content = ""


# # parse the queries
# query_path = 'Resources/query_desc.51-100.short.txt'
# queries = {}
# with open(query_path) as file:
#     for line in file:
#         # separate query number and query text
#         dot = line.find('.')
#         query_number = int(line[:dot].strip())
#         query_text = line[dot + 1:].strip()
#         query_text = re.sub(r'\b\.(?![a-zA-Z0-9])|\.\.+|[^\w\s.]', ' ', query_text)
#         # remove stopwords in query
#         queries[query_number] = remove_stopwords(query_text)

# go through each file and parse the documents
for filename in os.listdir(folder):
    if filename.lower() != 'readme':
        parse_file(os.path.join(folder, filename))
        ap89_num += 1

# get df and cf of term
term_df_file = "NonStemmed/term_df.json"
term_df = {}
term_cf_file = "NonStemmed/term_cf.json"
term_cf = {}


# write catalog file for each index file
def write_catalog(index_dict, index_file):
    index_file_root = index_file[8:-4]
    catalog_file_base = "Catalogs/catalog_{}.json"
    catalog_file = catalog_file_base.format(index_file_root)
    print("writing catalog", catalog_file)
    catalog = []
    init_offset = 0
    # write dict to binary file
    with open(index_file, 'wb') as bin_file:
        for term_id, doc_id in index_dict.items():
            # write as binary file and update offsets
            json_to_bytes = json.dumps({term_id: doc_id}).encode('utf-8')
            bin_file.write(json_to_bytes)
            curr_offset = bin_file.tell()
            size = curr_offset - init_offset
            catalog_line = (term_id, init_offset, size)
            catalog.append(catalog_line)
            init_offset = curr_offset
            # get df and cf of term
            freq = 0
            if term_id in term_df:
                term_df[term_id] += len(doc_id)
            else:
                term_df[term_id] = len(doc_id)
            for doc, positions in doc_id.items():
                freq += len(positions)
            if term_id in term_cf:
                term_cf[term_id] += freq
            else:
                term_cf[term_id] = freq

    # write catalog file
    with open(catalog_file, 'w') as json_file:
        json.dump(catalog, json_file, indent=2)


# go through 1000 documents at a time and create inverted index file
doc_num = 1
term_num = 1
index_file_base = "Indexes/index_{}.bin"
index_file = index_file_base.format(0)
index_dict = {}

doc_ids = {}
ids_doc = {}
term_ids = {}
ids_term = {}

for doc_no, text in doc_text_dict.items():
    # check with doc id dict
    if doc_no in doc_ids:
        doc_id = doc_ids[doc_no]
    else:
        doc_ids[doc_no] = doc_num
        ids_doc[doc_num] = doc_no
        doc_id = doc_ids[doc_no]

    # check if 1000 docs
    if doc_id % 1000 == 1:
        if doc_id > 1:
            write_catalog(index_dict, index_file)
        index_file = index_file_base.format(doc_id // 1000)
        index_dict = {}

    print("creating index...", doc_id, index_file)
    pos = 1
    for term in text:
        # check with term id dict
        if term in term_ids:
            term_id = term_ids[term]
        else:
            term_ids[term] = term_num
            ids_term[term_num] = term
            term_id = term_ids[term]
            term_num += 1

        # update/add to index file
        if term_id not in index_dict:
            index_dict[term_id] = {doc_id: [pos]}
        else:
            # update existing index
            if doc_id not in index_dict[term_id]:
                index_dict[term_id][doc_id] = [pos]
            else:
                # append pos, doc. already there
                index_dict[term_id][doc_id].append(pos)
        pos += 1
    doc_num += 1

# term/doc <-> id dictionaries
id_to_doc_file = "NonStemmed/id_to_doc.json"
doc_to_id_file = "NonStemmed/doc_to_id.json"
id_to_term_file = "NonStemmed/id_to_term.json"
term_to_id_file = "NonStemmed/term_to_id.json"
with open(id_to_doc_file, 'w') as json_file:
    json.dump(ids_doc, json_file, indent=2)
with open(doc_to_id_file, 'w') as json_file:
    json.dump(doc_ids, json_file, indent=2)
with open(term_to_id_file, 'w') as json_file:
    json.dump(term_ids, json_file, indent=2)
with open(id_to_term_file, 'w') as json_file:
    json.dump(ids_term, json_file, indent=2)

# create catalogs for each partial index
write_catalog(index_dict, index_file)

# get df and cf of terms
with open(term_df_file, 'w') as json_file:
    json.dump(term_df, json_file, indent=2)
with open(term_cf_file, 'w') as json_file:
    json.dump(term_cf, json_file, indent=2)

# counting number of terms in collections
vocab_size = len(ids_term)
print("# of unique terms in collection (VOCAB SIZE):", vocab_size)

# counting total number of tokens in the document collection
total_cf = sum(len(value) for value in doc_text_dict.values())
print("Total number of tokens (TOTAL CF):", total_cf)


# merge indexes and catalogs
def merge_files(catalog_file, index_file):
    print("merging files:", catalog_file, index_file)
    # read catalog to get offset/size info for index file
    with open(catalog_file, 'r') as json_file:
        catalog = json.load(json_file)
    with open(index_file, 'rb') as bin_file:
        for term, offset, size in catalog:
            # read binary
            bin_file.seek(offset)
            read_data = bin_file.read(size)
            index_data = json.loads(read_data)
            if term in final_index.keys():
                updated_data = {}
                for existing_key, existing_value in final_index[term].items():
                    updated_data[existing_key] = existing_value
                for term_id, doc_map in index_data.items():
                    for new_key, new_value in doc_map.items():
                        updated_data[new_key] = new_value
                final_index[term] = updated_data
            else:
                for value in index_data.values():
                    final_index[term] = value


final_index = {}
# for each catalog and corresponding index
for num, filename in enumerate(os.listdir("Catalogs")):
    merge_files(("Catalogs/catalog_index_" + str(num) + ".json"), ("Indexes/index_" + str(num) + ".bin"))


# final catalog
catalog_file = "final_catalog.json"
catalog = []
init_offset = 0
# write dict to binary file
with open("final_index.bin", 'wb') as bin_file:
    for term_id, doc_id in final_index.items():
        print("final catalog...", term_id)
        # write as binary file and update offsets
        # json_to_bytes = json.dumps({term_id: doc_id}).encode('utf-8')
        json_string = json.dumps({term_id: doc_id})
        json_string = json_string.replace('"', '').replace(" ", "")
        json_to_bytes = json_string.encode('utf-8')
        bin_file.write(json_to_bytes)
        curr_offset = bin_file.tell()
        size = curr_offset - init_offset
        catalog_line = (term_id, init_offset, size)
        catalog.append(catalog_line)
        init_offset = curr_offset
# write catalog file
with open(catalog_file, 'w') as json_file:
    json.dump(catalog, json_file, indent=2)

# for testing: hard code size to read from file
binary_filename = "Indexes/index_2.bin"
with open(binary_filename, "rb") as file:
    # go to start position using seek(offset)
    file.seek(0)  # test file
    # file.seek(194) # final
    # read(size of entry)
    bin_data = file.read(3464)  # test file
    # bin_data = file.read(1720) # final
    # decode
    data = bin_data.decode('utf-8')
print(data)
