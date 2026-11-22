
def fixed_size_chunking(text, chunk_size=500, overlap=50):
    """
    Divide text into fixed-size chunks with overlapping content.
    """
    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def paragraph_chunking(text, max_chunk_size=1000):
    """
    Group paragraphs into chunks without exceeding the preferred size.
    Oversized paragraphs are split into smaller pieces.
    """
    if not text or not text.strip():
        return []

    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero")

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        # Split paragraphs that are larger than the maximum size.
        if len(paragraph) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            for start in range(0, len(paragraph), max_chunk_size):
                chunks.append(
                    paragraph[start:start + max_chunk_size]
                )

            continue

        if not current_chunk:
            current_chunk = paragraph

        elif len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
            current_chunk += "\n\n" + paragraph

        else:
            chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
