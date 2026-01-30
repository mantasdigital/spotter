"""
Web Search Command Handlers.

Handles voice commands for web search and browsing.
Includes google search and URL browsing from v58.
"""

import re
import requests
from html.parser import HTMLParser
from html import unescape
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs, unquote

from hardware.interfaces import ILLMProvider


class HTMLTextExtractor(HTMLParser):
    """Simple HTML-to-text converter for summaries."""

    def __init__(self):
        super().__init__()
        self._chunks = []

    def handle_data(self, data):
        data = (data or "").strip()
        if data:
            self._chunks.append(data)

    def get_text(self) -> str:
        return " ".join(self._chunks).strip()


class WebSearchCommandHandler:
    """
    Handler for web search and browsing commands.

    Supports:
    - Google search: "google <query>", "search for <query>"
    - Browse URL: "browse <url>", "go to <url>"
    - Lithuanian variations: "paieško", "ieško"
    """

    def __init__(self, llm_provider: Optional[ILLMProvider] = None, language: str = "en"):
        """
        Initialize web search handler.

        Args:
            llm_provider: Optional LLM provider for summarization
            language: Current language ("en" or "lt")
        """
        self.llm = llm_provider
        self.language = language

    def can_handle(self, command: str) -> bool:
        """
        Check if this handler can process the command.

        Args:
            command: Voice command text

        Returns:
            bool: True if command is web search related
        """
        cmd_lower = command.lower().strip()

        # Google/search patterns
        if cmd_lower.startswith("google "):
            return True
        if " search " in cmd_lower or cmd_lower.startswith("search "):
            return True

        # Lithuanian search patterns
        if any(kw in cmd_lower for kw in ["paieško", "paieskok", "ieško", "ieskok"]):
            return True

        # Browse patterns
        if cmd_lower.startswith("browse "):
            return True
        if " go to " in cmd_lower or cmd_lower.startswith("go to "):
            return True

        return False

    def handle(self, command: str) -> Dict[str, Any]:
        """
        Handle web search/browse command.

        Args:
            command: Voice command text

        Returns:
            Dict with keys:
                - success (bool): Whether command was handled successfully
                - action (str): Action taken
                - message (str): Response message
                - query (str, optional): Search query
                - url (str, optional): URL to browse
        """
        cmd_lower = command.lower().strip()

        # Check for google/search
        if cmd_lower.startswith("google "):
            query = cmd_lower.replace("google ", "", 1).strip()
            return self._handle_google_search(query)

        if cmd_lower.startswith("search "):
            query = cmd_lower.replace("search ", "", 1).strip()
            if query.startswith("for "):
                query = query.replace("for ", "", 1).strip()
            return self._handle_google_search(query)

        # Lithuanian search
        if self.language == "lt" and any(kw in cmd_lower for kw in ["paieško", "paieskok", "ieško", "ieskok"]):
            # Extract query after the search keyword
            query = cmd_lower
            for kw in ["paieško", "paieskok", "ieško", "ieskok"]:
                query = query.replace(kw, "").strip()
            return self._handle_google_search(query)

        # Check for browse
        if cmd_lower.startswith("browse "):
            url = cmd_lower.replace("browse ", "", 1).strip()
            return self._handle_browse_url(url)

        if "go to " in cmd_lower or cmd_lower.startswith("go to "):
            url = cmd_lower.replace("go to ", "").strip()
            return self._handle_browse_url(url)

        return {
            "success": False,
            "action": "unknown",
            "message": "Web command not recognized"
        }

    def _handle_google_search(self, query: str) -> Dict[str, Any]:
        """
        Handle Google search command.

        Args:
            query: Search query

        Returns:
            Result dict with search information
        """
        if not query:
            return {
                "success": False,
                "action": "google_search",
                "message": "What should I search for?",
                "query": ""
            }

        try:
            # Perform DuckDuckGo search
            results = self._simple_search(query, max_results=3)

            if not results:
                return {
                    "success": False,
                    "action": "google_search",
                    "message": f"No results found for '{query}'",
                    "query": query
                }

            # Summarize results using LLM
            if self.llm:
                summary = self._summarize_search_results(query, results)
            else:
                # Fallback without LLM
                summary = f"Found {len(results)} results. " + results[0].get('snippet', '')[:200]

            return {
                "success": True,
                "action": "google_search",
                "message": summary,
                "query": query,
                "results": results
            }

        except Exception as e:
            return {
                "success": False,
                "action": "google_search",
                "message": f"Search error: {str(e)}",
                "query": query
            }

    def _handle_browse_url(self, url: str) -> Dict[str, Any]:
        """
        Handle browse URL command.

        Args:
            url: URL to browse

        Returns:
            Result dict with browse information
        """
        if not url:
            return {
                "success": False,
                "action": "browse_url",
                "message": "What website should I browse?",
                "url": ""
            }

        # Normalize URL
        cleaned_url = self._normalize_url(url)

        try:
            # Fetch and extract text from URL
            text = self._fetch_url_text(cleaned_url, max_chars=3000)

            if not text:
                return {
                    "success": False,
                    "action": "browse_url",
                    "message": f"Couldn't read anything useful from {cleaned_url}",
                    "url": cleaned_url
                }

            # Summarize using LLM
            if self.llm:
                summary = self._summarize_url_content(cleaned_url, text)
            else:
                # Fallback without LLM
                summary = text[:300] + "..."

            return {
                "success": True,
                "action": "browse_url",
                "message": summary,
                "url": cleaned_url
            }

        except Exception as e:
            return {
                "success": False,
                "action": "browse_url",
                "message": f"Error browsing {cleaned_url}: {str(e)}",
                "url": cleaned_url
            }

    def _normalize_url(self, url: str) -> str:
        """
        Normalize spoken URL to proper format.

        Args:
            url: Spoken URL (e.g., "w w w example dot com")

        Returns:
            str: Normalized URL (e.g., "www.example.com")
        """
        # Remove common filler words
        s = url.lower()
        replacements = {
            " dot ": ".",
            " slash ": "/",
            " colon ": ":",
            "w w w": "www",
            "h t t p": "http"
        }
        for k, v in replacements.items():
            s = s.replace(k, v)

        # Remove extra spaces
        s = re.sub(r"\s+", "", s)

        # Add http:// if not present
        if not s.startswith("http://") and not s.startswith("https://"):
            s = "http://" + s

        return s

    def _normalize_target_url(self, raw: str) -> str:
        """
        Take whatever comes from DuckDuckGo and return a clean https URL.
        Handles DuckDuckGo redirect wrappers (//duckduckgo.com/l/?uddg=...).

        Args:
            raw: Raw URL from search results

        Returns:
            str: Normalized URL or empty string if invalid
        """
        if not raw:
            return ""

        raw = raw.strip()

        # Handle protocol-relative URLs
        if raw.startswith("//"):
            raw = "https:" + raw

        # Decode HTML entities
        raw = unescape(raw)

        if raw.startswith("http://") or raw.startswith("https://"):
            url = raw
        else:
            url = "https://" + raw.lstrip("/")

        try:
            parsed = urlparse(url)

            # DuckDuckGo redirect wrapper: //duckduckgo.com/l/?uddg=ENCODED
            if parsed.netloc.endswith("duckduckgo.com") and parsed.path.rstrip("/") in ("/l", "/l/"):
                qs = parse_qs(parsed.query or "")
                uddg_list = qs.get("uddg")
                if uddg_list:
                    inner = unquote(uddg_list[0])
                    return self._normalize_target_url(inner)

            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return ""

            return url
        except Exception:
            return url

    def _simple_search(self, query: str, max_results: int = 3) -> list:
        """
        Perform DuckDuckGo search and extract results.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of dicts with 'title', 'url', 'snippet'
        """
        try:
            params = {"q": query}
            resp = requests.get(
                "https://duckduckgo.com/html/",
                params=params,
                timeout=8,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            html = resp.text
            results = []

            # Parse search results from HTML - try block pattern first
            block_pattern = r'<div class="result__body.*?">(.*?)</div>\s*</div>'
            for block in re.findall(block_pattern, html, flags=re.DOTALL):
                a_match = re.search(
                    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                    block,
                    flags=re.DOTALL,
                )
                if not a_match:
                    continue

                url_raw = a_match.group(1)
                title_raw = a_match.group(2)

                # Clean up title and URL
                title = re.sub(r"<.*?>", "", title_raw).strip()
                url = self._normalize_target_url(url_raw)

                if not title or not url:
                    continue

                # Extract snippet
                snippet_match = re.search(
                    r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
                    block,
                    flags=re.DOTALL,
                )
                snippet = ""
                if snippet_match:
                    snippet = re.sub(r"<.*?>", "", snippet_match.group(1)).strip()

                results.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet
                })

                if len(results) >= max_results:
                    break

            # Fallback pattern if block pattern didn't find anything
            # DuckDuckGo may change HTML structure, so we have a backup
            if not results:
                # Look for result links directly
                link_pattern = r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
                for m in re.finditer(link_pattern, html, flags=re.DOTALL):
                    url_raw = m.group(1)
                    title_html = m.group(2)

                    title = re.sub(r"<.*?>", "", title_html).strip()
                    url = self._normalize_target_url(url_raw)

                    if not title or not url:
                        continue

                    results.append({
                        "title": title,
                        "url": url,
                        "snippet": ""
                    })

                    if len(results) >= max_results:
                        break

                # Try to find snippets separately and match them
                if results:
                    snippet_pattern = r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>'
                    snippets = []
                    for m in re.finditer(snippet_pattern, html, flags=re.DOTALL):
                        snippet_html = m.group(1)
                        snippet = re.sub(r"<.*?>", "", snippet_html).strip()
                        if snippet:
                            snippets.append(snippet)

                    # Assign snippets to results (they should be in same order)
                    for i, snippet in enumerate(snippets):
                        if i < len(results):
                            results[i]["snippet"] = snippet

            return results

        except Exception as e:
            print(f"[SEARCH] Error: {e}")
            return []

    def _fetch_url_text(self, url: str, max_chars: int = 3000) -> str:
        """
        Fetch URL and extract visible text.

        Args:
            url: URL to fetch
            max_chars: Maximum characters to extract

        Returns:
            str: Extracted text content
        """
        try:
            resp = requests.get(
                url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (TARS-PiCarX)"},
            )

            if not (200 <= resp.status_code < 300):
                print(f"[BROWSE] HTTP {resp.status_code} for {url}")
                return ""

            html = resp.text

            # Extract visible text
            parser = HTMLTextExtractor()
            parser.feed(html or "")
            text = parser.get_text()

            # Clean up whitespace
            text = re.sub(r"\s+", " ", text).strip()

            if len(text) > max_chars:
                text = text[:max_chars]

            return text.strip()

        except Exception as e:
            print(f"[BROWSE] Error: {e}")
            return ""

    def _summarize_search_results(self, query: str, results: list) -> str:
        """
        Use LLM to summarize search results.

        Args:
            query: Original search query
            results: List of search result dicts

        Returns:
            str: Summarized response
        """
        if not self.llm or not results:
            return ""

        try:
            # Build context from results
            context = f"Search query: {query}\n\nTop results:\n"
            for i, result in enumerate(results[:3], 1):
                context += f"\n{i}. {result.get('title', 'No title')}\n"
                context += f"   {result.get('snippet', 'No description')}\n"

            # Ask LLM to summarize
            messages = [
                {
                    "role": "system",
                    "content": "You are TARS. Summarize search results concisely in 2-3 sentences with wit."
                },
                {
                    "role": "user",
                    "content": f"Summarize these search results for '{query}':\n{context}"
                }
            ]

            response = self.llm.chat(messages=messages, max_tokens=150)
            summary = self.llm.extract_text(response)

            return summary.strip()

        except Exception as e:
            print(f"[SEARCH] Summarization error: {e}")
            # Fallback to first snippet
            return results[0].get('snippet', 'Found results.')[:200]

    def _summarize_url_content(self, url: str, text: str) -> str:
        """
        Use LLM to summarize URL content.

        Args:
            url: URL that was browsed
            text: Extracted text content

        Returns:
            str: Summarized content
        """
        if not self.llm:
            return text[:300]

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are TARS. Summarize web content concisely in 2-3 sentences."
                },
                {
                    "role": "user",
                    "content": f"Summarize this content from {url}:\n\n{text[:2000]}"
                }
            ]

            response = self.llm.chat(messages=messages, max_tokens=150)
            summary = self.llm.extract_text(response)

            return summary.strip()

        except Exception as e:
            print(f"[BROWSE] Summarization error: {e}")
            return text[:300] + "..."
