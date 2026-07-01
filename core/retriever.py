from core.db import get_vector_store

def get_context(query: str, k: int = 5) -> tuple[str, list[dict]]:
    """
    Searches the active vector database for text chunks matching the query.
    Formats the matching chunks into a cohesive context string labeled with sources,
    and returns a list of individual chunk metadata for citation rendering.
    
    Returns:
        formatted_context (str): The concatenated text for the LLM prompt.
        citations (list[dict]): Raw metadata representing each source snippet.
    """
    # 1. Connect to active Vector DB
    store = get_vector_store()
    
    # 2. Retrieve top-k documents
    results = store.similarity_search(query, k=k)
    
    if not results:
        return "No relevant literature chunks found in database.", []
        
    # 3. Format context text blocks with explicit labeling
    context_blocks = []
    citations = []
    
    for idx, item in enumerate(results):
        text = item["text"]
        meta = item["metadata"]
        
        # Build clean source label
        source_name = meta.get("source", "Unknown Source")
        doc_type = meta.get("type", "unknown")
        
        if doc_type == "pdf":
            page = meta.get("page", "unknown")
            label = f"Document: {source_name} (Page {page})"
        elif doc_type == "epub":
            section = meta.get("section", "unknown")
            label = f"Document: {source_name} (Section: {section})"
        elif doc_type == "web":
            title = meta.get("title", source_name)
            label = f"Website: {title} ({source_name})"
        else:
            label = f"Source: {source_name}"
            
        context_blocks.append(f"--- [Snippet #{idx+1} | {label}] ---\n{text}\n")
        
        citations.append({
            "index": idx + 1,
            "label": label,
            "source": source_name,
            "type": doc_type,
            "metadata": meta,
            "snippet": text[:200] + "..." if len(text) > 200 else text
        })
        
    formatted_context = "\n".join(context_blocks)
    
    return formatted_context, citations
