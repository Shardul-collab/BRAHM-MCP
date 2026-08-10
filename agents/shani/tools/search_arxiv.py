import requests
import xml.etree.ElementTree as ET
import time


ARXIV_API = "http://export.arxiv.org/api/query"

HEADERS = {
    "User-Agent": "SHANI/1.0 (Research Bot)"
}


def search_arxiv(query, max_results=25, retries=3):

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }

    for attempt in range(retries + 1):

        try:
            response = requests.get(
                ARXIV_API,
                params=params,
                headers=HEADERS,
                timeout=15
            )

            if response.status_code != 200:
                print(f"[arXiv WARNING] Status {response.status_code}")
                time.sleep(2)
                continue

            try:
                root = ET.fromstring(response.text)
            except Exception as e:
                print(f"[arXiv XML ERROR] {e}")
                time.sleep(2)
                continue

            papers = []

            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):

                try:
                    title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                    summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                    id_elem = entry.find("{http://www.w3.org/2005/Atom}id")

                    if title_elem is None:
                        continue

                    title = title_elem.text.strip().replace("\n", " ")
                    summary = summary_elem.text.strip() if summary_elem is not None else ""

                    # 🔥 ROBUST PDF EXTRACTION
                    pdf_url = None

                    for l in entry.findall("{http://www.w3.org/2005/Atom}link"):
                        href = l.attrib.get("href", "")

                        if "pdf" in href:
                            pdf_url = href
                            break

                    # fallback: construct from ID
                    if not pdf_url and id_elem is not None:
                        paper_id = id_elem.text.split("/")[-1]
                        pdf_url = f"https://arxiv.org/pdf/{paper_id}.pdf"

                    papers.append({
                        "title": title,
                        "summary": summary,
                        "source": "arxiv",
                        "pdf_url": pdf_url
                    })

                except Exception as e:
                    print(f"[arXiv ENTRY ERROR] {e}")
                    continue

            print(f"[S2] arXiv success: {len(papers)} papers")
            return papers

        except Exception as e:
            print(f"[arXiv ERROR] attempt {attempt}: {e}")
            time.sleep(2)

    print("[arXiv] failed after retries")
    return []