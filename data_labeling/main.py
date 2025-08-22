from __future__ import annotations

from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from label_api.lstudio_interfacer import Labeller
from langchain_openai import ChatOpenAI 
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from vector_db import VectorDb
import json
from typing import Any, Dict, Optional, Tuple

# Queue helpers (use these instead of any local Redis calls)
from queue_helpers import (
    enqueue_paper_id,
    pop_paper_id,             # simple pattern
    claim_next_paper,         # safer pattern (claim + ack/requeue)
    ack_paper,
    requeue_inflight,
    push_completed_annotation,
)


# --------------------
# RAG / Model setup
# --------------------

vdb = VectorDb()
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = PineconeVectorStore(
    index=vdb.__get_index__(), embedding=embedder, namespace="article_upload_test_1"
)

llm = ChatOpenAI(model="gpt-4o")
prompt = hub.pull("rlm/rag-prompt")

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vector_store.as_retriever(),
    chain_type_kwargs={"prompt": prompt},
)

prompts = [
    "What part of the paper talk about sample size ....",
    "etc...",
]

LS = Labeller()
app = FastAPI()

LS.create_webhook(
    endpoint="http://localhost:8000"
) 

# --------------------
# Schemas
# --------------------

class CriteriaRequest(BaseModel):
    criteria: str


# --------------------
# Webhooks 
# --------------------

@app.post("/webhook")
async def ls_webhook(req: Request, bg: BackgroundTasks):
    payload = await req.json()

    print("WEBHOOK RECEIVED  ")
    # print(payload.get("event", "no_event"))
    print(json.dumps(payload, indent=2))
    action = payload.get("action")

    if  action == "PROJECT_CREATED":
        # If a new project is created, we can start importing tasks
        proj_id = payload["project"]["id"]
        bg.add_task(import_next_paper_tasks, proj_id)
        return {"status": "ok", "event": "project_created"}

        # return {"status": "ignored", "event": payload.get("event")}

    proj_id = payload["project"]["id"]

    # if  action == "TASK_CREATED":
    if  action == "ANNOTATION_COMPLETED":
        task_info = payload["tasks"]
        annotation = payload["annotation"]
        bg.add_task(handle_completed_task, task_info, annotation)
        return {"status": "ok", "event": "annotation_completed"}

    # Handle the completed task in the background
    # bg.add_task(handle_completed_task, task_info, annotation)e

    # If there are no new tasks left in the project, pull the next paper from the queue
    new_count = LS.count_new_tasks(proj_id)
    if new_count == 0:
        bg.add_task(import_next_paper_tasks, proj_id)

    return {"status": "ok"}


# --------------------
# Queue-driven import
# --------------------

USE_SAFE_QUEUE = True  # toggle to False to use simple pop() pattern


def _unpack_claim(claim: Any) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort unpack for different possible claim shapes.
    Returns (paper_id, claim_token).
    """
    if claim is None:
        return None, None
    # Dict shape
    if isinstance(claim, dict):
        pid = (
            claim.get("paper_id")
            or claim.get("id")
            or claim.get("item_id")
            or claim.get("article_id")
        )
        token = claim.get("claim_id") or claim.get("receipt") or claim.get("token")
        return pid, token
    # Tuple or list shape
    if isinstance(claim, (tuple, list)) and len(claim) >= 2:
        return str(claim[0]) if claim[0] is not None else None, str(claim[1]) if claim[1] is not None else None
    # Str shape (just an id, no claim token)
    if isinstance(claim, str):
        return claim, None
    return None, None


def import_next_paper_tasks(project_id: int) -> None:
    """Pull the next paper from the queue and create Label Studio tasks.

    Uses the safer claim/ack pattern when available, with a fallback to simple pop.
    Creates a paper-specific RAG pipeline to ensure responses only come from the relevant paper.
    Retrieves ALL chunks for the paper to allow the LLM to reason over the complete content.
    """
    paper_id: Optional[str] = None
    claim_token: Optional[str] = None

    if USE_SAFE_QUEUE:
        claim = claim_next_paper()
        paper_id, claim_token = _unpack_claim(claim)
    else:
        paper_id = pop_paper_id()

    if not paper_id:
        return

    try:
        tasks = []
        # Create a retriever with metadata filter for this specific paper
        # and increase the number of results to get all chunks
        filtered_retriever = vector_store.as_retriever(
            search_kwargs={
                "filter": {"doc": paper_id},  # Filter to only search within this paper
                "k": 100  # Increase this if papers have more chunks
            }
        )
        
        # Create a new QA chain with the filtered retriever
        paper_qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=filtered_retriever,
            return_source_documents=True,
            chain_type="stuff",  # Explicitly set to use stuff chain
            chain_type_kwargs={
                "prompt": prompt
            }
        )

        for q in prompts:
            result = paper_qa_chain.invoke({
                "query": q,
                "context": "Consider ALL provided chunks of the paper when answering. Synthesize information from all relevant sections."
            })
            llm_answer = result.get("result") if isinstance(result, dict) else result

            tasks.append(
                {
                    "data": {
                        "paper_id": paper_id,
                        "title": "SOME TITLE | REPLACE LATER",
                        "paper_text": llm_answer or "NO LLM ANSWER",
                        "class_criteria": q,
                    }
                }
            )

        LS.import_task(tasks)

        # Acknowledge the claimed item only if we used the claim pattern
        if claim_token:
            ack_paper(claim_token)

    except Exception:
        # If something failed after claiming, requeue the inflight item
        if claim_token:
            requeue_inflight(claim_token)
        raise


# --------------------
# Completed task handling
# --------------------

def handle_completed_task(task: Dict[str, Any], annotation: Dict[str, Any]) -> None:
    record: Dict[str, Any] = {}

    for prefix, d in (("task", task), ("ann", annotation)):
        for k, v in d.items():
            record[f"data_{k}"] = v

    # Persist the completed annotation in the queue-backed sink
    push_completed_annotation(json.dumps(record))
