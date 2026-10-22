import httpx


async def post_model_generation(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    request_id: str,
    citation_attempt: int,
    max_new_tokens: int = 768,
    timeout_seconds: float = 180.0,
) -> httpx.Response:
    """Send the assembled prompt unchanged using the Model API contract."""
    response = await client.post(
        url,
        json={"prompt": prompt, "max_new_tokens": max_new_tokens},
        headers={
            "X-Request-ID": request_id,
            "X-Citation-Attempt": str(citation_attempt),
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    return response
