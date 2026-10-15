"""
PoC: Decision loop prototype using decision models in Ollama
and Playwright browser automation.
"""

import asyncio
from playwright.async_api import async_playwright
import ollama


class DecisionRouter:
    def __init__(self, model_name: str = "nimble"):
        self.model = model_name

    async def decide_binary(self, state_text: str, question: str, true_crit: str, false_crit: str) -> bool:
        """
        Returns a fast True/False decision using the System One endpoint.
        """
        try:
            res = ollama.systemone(
                model=self.model,
                state=state_text,
                questions={
                    "decision": {
                        "type": "noul",
                        "instructions": question,
                        "criteria": {
                            "true": true_crit,
                            "false": false_crit
                        }
                    }
                }
            )
            return res.answers["decision"]["value"]
        except Exception:
            # Fallback for standard ollama chat interface with JSON format enforcement
            res = ollama.chat(
                model="llama3.2:1b",
                messages=[
                    {"role": "user", "content": f"Evaluate state: {state_text}\nQuestion: {question}"}
                ],
                format={
                    "type": "object",
                    "properties": {"decision": {"type": "boolean"}},
                    "required": ["decision"]
                },
                options={"temperature": 0}
            )
            import json
            return json.loads(res["message"]["content"]).get("decision", False)


async def main():
    router = DecisionRouter()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://news.ycombinator.com")

        # Retrieve simplified page text
        content = await page.inner_text("body")
        sample_state = content[:1500]

        print("[System 1] Inspecting page for visible login form...")
        is_login = await router.decide_binary(
            state_text=sample_state,
            question="Is there currently a visible login form displayed on the page?",
            true_crit="Username and password fields for input are visible.",
            false_crit="This is a regular page with an article/post list."
        )
        print(f"-> Fast decision result: {is_login}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
