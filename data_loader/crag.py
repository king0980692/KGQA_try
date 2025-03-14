import json
import bz2
from tqdm import tqdm
from bs4 import BeautifulSoup


def get_multi_hop_questions(
    top_n: int = None, line_id: int = None
) -> tuple[list[int], dict[int, dict[str, any]]]:
    CRAG_DATA_PATH = "./data/crag_task_1_dev_v4_release.jsonl.bz2"
    MULTIHOP_DATA_PATH = "./data/multi-hops.txt"

    if top_n is not None:
        # get all multi-hops question line ids
        with open(MULTIHOP_DATA_PATH, "r") as f:
            global multi_hop_infos
            multi_hop_infos = json.load(f)

        multi_hop_line_ids = [qa_info["line_id"]
                              for qa_info in multi_hop_infos[:top_n]]
    elif line_id is not None:
        multi_hop_line_ids = [line_id]

    # get multi-hops questions
    multi_hop_questions = {}
    with bz2.open(CRAG_DATA_PATH, "rt") as f:
        for idx, line in tqdm(enumerate(f), total=max(multi_hop_line_ids)):
            if idx in multi_hop_line_ids:
                multi_hop_questions[idx] = json.loads(line)

                if idx == max(multi_hop_line_ids):
                    break

    return multi_hop_line_ids, multi_hop_questions


def is_existed_page(page_contents: list[str], page_content: str) -> bool:
    if page_content in page_contents:
        return True
    return False


def extract_page_content(page_origin_content: str) -> str:
    soup = BeautifulSoup(page_origin_content, "html.parser")
    page_content = soup.get_text(" ", strip=True)
    return page_content


def get_page_contents(
    multi_hop_line_ids: list[int], multi_hop_questions: dict[int, dict[str, any]]
) -> dict[int, list[dict[str, str]]]:
    page_contents = {line_id: [] for line_id in multi_hop_line_ids}

    for line_id in tqdm(multi_hop_line_ids):
        search_results = multi_hop_questions[line_id]["search_results"]

        for search_result in search_results:  # 5
            # get page content
            page_name = search_result["page_name"]
            page_origin_content = search_result["page_result"]
            page_content = extract_page_content(page_origin_content)

            # pair page name and content
            if len(page_content) > 0 and not is_existed_page(
                page_contents[line_id], page_content
            ):
                page_contents[line_id].append(
                    {"page_name": page_name, "page_content": page_content}
                )

    return page_contents


def load_crag_pages(
    top_n: int = None, line_id: int = None
) -> dict[int, list[dict[str, str]]]:
    multi_hop_line_ids, multi_hop_questions = get_multi_hop_questions(
        top_n, line_id)
    page_contents_dict = get_page_contents(
        multi_hop_line_ids, multi_hop_questions)
    return page_contents_dict
