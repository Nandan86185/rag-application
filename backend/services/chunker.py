def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    
    if not text:
        return []
        
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If we reached the end of the text, append the remainder and break
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
            
        # Get the current window
        chunk = text[start:end]
        
        # Try to find a clean breaking point (paragraph, newline, or sentence)
        break_point = -1
        for sep in ["\n\n", "\n", ". "]:
            idx = chunk.rfind(sep)
            if idx != -1:
                # We found a natural break point inside the chunk limit
                break_point = idx + len(sep)
                break
                
        if break_point != -1:
            end = start + break_point
            
        chunks.append(text[start:end].strip())
        
        # Advance the start pointer, accounting for the desired overlap
        start = end - overlap
        
    # Remove any empty chunks
    return [c for c in chunks if c]
