import os
import re
import json
from json import JSONDecodeError
from tqdm import tqdm
import logging
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from gliner import GLiNER
from litellm import completion
import networkx as nx
from pyvis.network import Network
from data_loader.wiki import load_wikipedia_page
from data_loader.crag import load_crag_pages


def create_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def chunk_page_content(page_content: str, path: str) -> list[str]:
    if os.path.exists(path):
        logging.info("load chunks from file")

        with open(path, "r") as f:
            chunks = json.load(f)
    else:
        logging.info("chunk the page content")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600, chunk_overlap=50, separators=["\n\n", "\n", ". "]
        )
        chunks = text_splitter.split_text(page_content)

        with open(path, "w") as f:
            json.dump(chunks, f, indent=4)
    return chunks


def merge_entities(text: str, entities: list[dict[str, any]]) -> list[dict[str, any]]:
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity["label"] == current["label"] and (
            next_entity["start"] == current["end"] + 1
            or next_entity["start"] == current["end"]
        ):
            current["text"] = text[current["start"]: next_entity["end"]].strip()
            current["end"] = next_entity["end"]
        else:
            merged.append(current)
            current = next_entity

    # Append the last entity
    merged.append(current)
    return merged


def extract_entities(
    chunks: list[str],
    labels: list[str],
    entity_list_path: str = None,
    chunks_entities_path: str = None,
) -> tuple[list[str], list[list[str]]]:
    """by NER model"""
    if (
        entity_list_path
        and chunks_entities_path
        and os.path.exists(entity_list_path)
        and os.path.exists(chunks_entities_path)
    ):
        logging.info("load entities from file")

        with open(entity_list_path, "r") as f:
            entity_list = json.load(f)
        with open(chunks_entities_path, "r") as f:
            chunks_entities = json.load(f)
    else:
        logging.info("extract entities from chunks")

        model = GLiNER.from_pretrained("numind/NuNerZero")
        entity_list = []
        chunks_entities = []
        duplicates = set()

        for text in tqdm(chunks):
            entities = model.predict_entities(text, labels, threshold=0.7)
            entities = merge_entities(text, entities)
            chunk_entities = set()

            for entity in entities:
                entity["text"] = entity["text"].lower()
                chunk_entities.add(entity["text"])
                if entity["text"] in duplicates:
                    continue
                duplicates.add(entity["text"])
                entity_list.append((entity["text"], "=>", entity["label"]))

            chunks_entities.append(list(chunk_entities))

        if entity_list_path and chunks_entities_path:
            with open(entity_list_path, "w") as f:
                json.dump(entity_list, f, indent=4)
            with open(chunks_entities_path, "w") as f:
                json.dump(chunks_entities, f, indent=4)
    return entity_list, chunks_entities


def classify_entities(entity_list: list[str], labels: list[str]) -> dict[str, set[str]]:
    labels_entities = {label: set() for label in labels}

    for e in entity_list:
        s, _, o = e
        labels_entities[o].add(s)

    return labels_entities


def extract_json_list(response: str) -> list[str]:
    json_str_list = re.findall("{.+}", response)
    return [json.loads(json_str) for json_str in json_str_list]


def run_llm(messages: list[dict[str, str]]) -> str:
    response = (
        completion(
            model="ollama/phi4",
            messages=messages,
            api_base="http://140.119.164.60:11434",
            max_tokens=1000,
        )
        .choices[0]
        .message.content
    )
    logging.info(messages[1]["content"])
    logging.info(response)
    return response


def extract_triples(
    chunks: list[str],
    chunks_entities: list[list[str]],
    triples_path: str,
    errors_path: str,
) -> tuple[list[list[str]], list[str]]:
    """given entities output triples -- by LLM"""
    if os.path.exists(triples_path) and os.path.exists(errors_path):
        logging.info("load triples from file")

        with open(triples_path, "r") as f:
            chunks_triples = json.load(f)
        with open(errors_path, "r") as f:
            errors = json.load(f)
    else:
        logging.info("extract triples from chunks")

        system_message = """Extract all the relationships between the following entities ONLY based on the given ones.
        Return a list of JSON objects. For example:

        <Examples>
            [{{"subject": "John", "relationship": "lives in", "object": "US"}},
            {{"subject": "Eifel towel", "relationship": "is located in", "object": "Paris"}},
            {{"subject": "Hayao Miyazaki", "relationship": "is", "object": "Japanese animator"}}]
        </Examples>

        Note:
        1. ONLY return triples and nothing else. 
        2. None of "subject", "relationship" and "object" can be empty. 
        3. "Subject" and "object" should be a string and if it appears in entities list, please follow the text used in entities list.
        4. If many entities match, please write them separately into several triples.

        Entities: \n\n{entities}"""

        user_message = "Context: {text}\n\nTriples:"

        errors = []
        chunks_triples = []

        for i in tqdm(range(len(chunks_entities))):
            try:
                text = chunks[i]
                ents = "\n\n".join(chunks_entities[i])

                response = run_llm(
                    [
                        {
                            "content": system_message.format(entities=ents),
                            "role": "system",
                        },
                        {"content": user_message.format(
                            text=text), "role": "user"},
                    ]
                )

                triples = extract_json_list(response)
                additional_triples = []
                for idx, triple in enumerate(triples):
                    for key in triple:
                        print(triples[idx])
                        if triples[idx][key] is None:
                            continue
                        elif isinstance(triples[idx][key], list):
                            for item in triples[idx][key]:
                                additional_triples.append(
                                    {
                                        "subject": triple["subject"],
                                        "relationship": triple["relationship"],
                                        "object": item.lower(),
                                    }
                                )
                        else:
                            triples[idx][key] = triples[idx][key].lower()
                chunks_triples.append(triples + additional_triples)

                logging.info(f"Chunks: {text}")
                logging.info(f"Entities: {'; '.join(chunks_entities[i])}")
                logging.info(f"Triples: {triples}")
            except JSONDecodeError as e:
                errors.append(text)
                chunks_triples.append([])

                logging.error(f"Chunks: {text}")
                logging.error(f"{e} in chunk {i}")

        with open(triples_path, "w") as f:
            json.dump(chunks_triples, f, indent=4)
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=4)

    return chunks_triples, errors


def run_construction_once(
    page_content: str,
    construction_dir: str,
    labels: list[str],
    idx: int = 0,
) -> tuple[list[list[str]], dict[str, set[str]]]:
    CHUNKS_PATH = f"{construction_dir}/chunks{idx}.txt"
    chunks = chunk_page_content(page_content, CHUNKS_PATH)
    logging.info(f"# of chunks: {len(chunks)}")

    ENTITY_LIST_PATH = f"{construction_dir}/entities{idx}.txt"
    CHUNKS_ENTITIES_PATH = f"{construction_dir}/chunks_entities{idx}.txt"
    entity_list, chunks_entities = extract_entities(
        chunks, labels, ENTITY_LIST_PATH, CHUNKS_ENTITIES_PATH
    )
    logging.info(f"# of entities: {len(entity_list)}")

    labels_entities = classify_entities(entity_list, labels)
    logging.info({label: len(entities)
                 for label, entities in labels_entities.items()})

    TRIPLES_PATH = f"{construction_dir}/chunks_triples{idx}.txt"
    ERRORS_PATH = f"{construction_dir}/errors{idx}.txt"
    chunks_triples, _ = extract_triples(
        chunks, chunks_entities, TRIPLES_PATH, ERRORS_PATH
    )
    logging.info(
        f"# of triples: {sum(len(triples) for triples in chunks_triples)}")

    return chunks_triples, labels_entities


def get_color(label: str, labels_entities: dict[str, list[str]]) -> str:
    colors = [
        "orange",
        "blue",
        "green",
        "brown",
        "red",
        "purple",
        "yellow",
        "pink",
        "cyan",
        "magenta",
        "lime",
        "teal",
        "navy",
        "gold",
        "silver",
        "violet",
        "coral",
        "indigo",
        "salmon",
        "turquoise",
        "black",
    ]
    for idx, label_entities in enumerate(list(labels_entities.values())):
        if label in label_entities:
            return colors[idx]
    return colors[-1]


def get_size(label: str, labels_entities: dict[str, list[str]]) -> int:
    sizes = list(range(40, 0, -2))
    for idx, label_entities in enumerate(list(labels_entities.values())):
        if label in label_entities:
            return sizes[idx]
    return sizes[-1]


def draw_graph(
    chunks_triples: list[list[str]],
    labels_entities: dict[str, list[str]],
    path: str = None,
) -> nx.Graph:
    G = nx.Graph()

    for items in chunks_triples:
        for item in items:
            try:
                node1 = item["subject"]
                node2 = item["object"]
                G.add_node(
                    node1,
                    title=str(node1),
                    color=get_color(node1, labels_entities),
                    size=get_size(node1, labels_entities),
                    label=str(node1),
                )
                G.add_node(
                    node2,
                    title=str(node2),
                    color=get_color(node2, labels_entities),
                    size=get_size(node2, labels_entities),
                    label=str(node2),
                )
                G.add_edge(
                    node1,
                    node2,
                    title=str(item["relationship"]),
                    weight=4,
                    head=str(node1),
                    tail=str(node2),
                )
            except Exception:
                logging.error(f"Error in item: {item}")

    if path:
        nt = Network(height="750px", width="100%")
        nt.from_nx(G)
        nt.force_atlas_2based(central_gravity=0.015, gravity=-31)
        nt.save_graph(path)

    return G


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--dataset", type=str, required=True, choices=["crag", "wiki"]
    )
    arg_parser.add_argument("--wiki_title", type=str)
    arg_parser.add_argument("--crag_top_n", type=int)
    arg_parser.add_argument("--crag_line_id", type=int)
    args = arg_parser.parse_args()

    # check whether the arguments are valid
    if args.dataset == "crag":
        if args.crag_top_n is None and args.crag_line_id is None:
            raise ValueError(
                "Please provide either crag_top_n or crag_line_id")
    elif args.dataset == "wiki":
        if args.wiki_title is None:
            raise ValueError("Please provide wiki_title")

    # set labels
    LABELS = [
        "person",
        "organization",
        "location",
        "event",
        "date",
        "product",
        "law",
        "medical",
        "scientific_term",
        "work_of_art",
        "language",
        "nationality",
        "religion",
        "sport",
        "weapon",
        "food",
        "currency",
        "disease",
        "animal",
        "plant",
    ]

    global chunks_triples, labels_entities
    chunks_triples = []
    labels_entities = {}

    if args.dataset == "crag":
        # load crag pages
        page_contents_dict = load_crag_pages(
            args.crag_top_n, args.crag_line_id)

        for line_id, page_contents in page_contents_dict.items():
            # paths
            TITLE = f"question_{line_id}"
            TITLE_DIR = f"./question_{line_id}"
            CONSTRUCTION_DIR = f"{TITLE_DIR}/construction"
            LOG_DIR = f"{TITLE_DIR}/log"
            create_dir(TITLE_DIR)
            create_dir(CONSTRUCTION_DIR)
            create_dir(LOG_DIR)

            # logging
            logging.basicConfig(
                level=logging.INFO,
                datefmt="%Y-%m-%d %H:%M:%S",
                format="%(asctime)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler(
                        f"{LOG_DIR}/construction.log",
                        mode="w",
                    ),
                    logging.StreamHandler(),
                ],
                force=True,
            )

            for idx, page_name_content in enumerate(page_contents):
                page_name = page_name_content["page_name"]
                page_content = page_name_content["page_content"]

                # construction
                chunks_triples_part, labels_entities_part = run_construction_once(
                    page_content, CONSTRUCTION_DIR, LABELS, idx
                )
                chunks_triples.extend(chunks_triples_part)
                for label, entities in labels_entities_part.items():
                    if label not in labels_entities:
                        labels_entities[label] = set(entities)
                    else:
                        for entity in entities:
                            labels_entities[label].add(entity)

            # save graph
            GRAPH_PATH = f"{CONSTRUCTION_DIR}/graph.html"
            draw_graph(chunks_triples, labels_entities, GRAPH_PATH)
    elif args.dataset == "wiki":
        # paths
        TITLE = args.wiki_title
        TITLE_DIR = f"./{''.join(TITLE.split(' '))}"
        CONSTRUCTION_DIR = f"{TITLE_DIR}/construction"
        LOG_DIR = f"{TITLE_DIR}/log"
        create_dir(TITLE_DIR)
        create_dir(CONSTRUCTION_DIR)
        create_dir(LOG_DIR)

        # logging
        logging.basicConfig(
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(
                    f"{LOG_DIR}/construction.log",
                    mode="w",
                ),
                logging.StreamHandler(),
            ],
            force=True,
        )

        # load wikipedia page
        page_content = load_wikipedia_page(
            TITLE, f"{CONSTRUCTION_DIR}/page_content.txt"
        )

        # construction
        chunks_triples, labels_entities = run_construction_once(
            page_content, CONSTRUCTION_DIR, LABELS
        )

        # save graph
        GRAPH_PATH = f"{CONSTRUCTION_DIR}/graph.html"
        draw_graph(chunks_triples, labels_entities, GRAPH_PATH)
