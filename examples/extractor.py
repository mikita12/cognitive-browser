"""
DOM Extractor & Candidate Selector for Playwright.
Parses page interactive elements into lightweight, token-efficient candidate structures.
"""

from typing import List, Dict, Any
from playwright.async_api import Page


class DomCandidateExtractor:
    """
    Interactive element candidate extractor.
    Replaces 50,000+ characters of raw HTML with a compact vector of interactive candidates
    with minimal token overhead (~20-50 tokens per candidate).
    """

    EXTRACT_JS = r"""
    () => {
        const candidates = [];
        const isVisible = (el) => {
            const rect = el.getBoundingClientRect();
            const style = window.getComputedStyle(el);
            return rect.width > 0 && rect.height > 0 && 
                   style.visibility !== 'hidden' && 
                   style.display !== 'none' && 
                   style.opacity !== '0';
        };

        const getCleanText = (el) => {
            return (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('placeholder') || el.getAttribute('value') || '')
                   .replace(/\s+/g, ' ')
                   .trim();
        };

        const interactiveSelectors = 'button, a, input, select, textarea, [role="button"], [role="link"], [role="tab"], [role="checkbox"]';
        const elements = Array.from(document.querySelectorAll(interactiveSelectors));

        elements.forEach((el, index) => {
            if (!isVisible(el)) return;

            const tag = el.tagName.toLowerCase();
            const type = el.getAttribute('type') || '';
            let defaultRole = tag;
            if (type === 'submit') defaultRole = 'submit button';
            else if (tag === 'input') defaultRole = `${type || 'text'} input`;

            const role = el.getAttribute('role') || defaultRole;
            const text = getCleanText(el);
            const id = el.id || '';
            const name = el.getAttribute('name') || '';

            // Skip empty decorative elements
            if (!text && !id && !name && tag !== 'input') return;

            // Generate unique reference attribute for Playwright targeting
            const ref = `s1-ref-${index}`;
            el.setAttribute('data-s1-ref', ref);

            candidates.push({
                ref: ref,
                tag: tag,
                role: role,
                text: text.slice(0, 100),
                type: type,
                id: id,
                name: name
            });
        });

        return candidates;
    }
    """

    @classmethod
    async def extract_interactive_elements(cls, page: Page, max_elements: int = 40) -> List[Dict[str, Any]]:
        """
        Retrieves list of interactive elements visible in viewport.
        """
        elements = await page.evaluate(cls.EXTRACT_JS)
        return elements[:max_elements]

    @classmethod
    def format_for_prompt(cls, candidates: List[Dict[str, Any]]) -> str:
        """
        Compact formatting of candidate elements for decision models.
        Payload: ~10 tokens per element.
        """
        lines = []
        for c in candidates:
            parts = [f"[{c['ref']}] {c['role']}"]
            if c['text']:
                parts.append(f'"{c["text"]}"')
            if c['type']:
                parts.append(f"type={c['type']}")
            lines.append(" | ".join(parts))
        return "\n".join(lines)
