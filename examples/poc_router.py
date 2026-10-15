"""
PoC: Prototyp pętli decyzyjnej z wykorzystaniem modelu decyzyjnego w Ollamie
oraz automatyzacji Playwright.
"""

import asyncio
from playwright.async_api import async_playwright
import ollama

class DecisionRouter:
    def __init__(self, model_name: str = "nimble"):
        self.model = model_name

    async def decide_binary(self, state_text: str, question: str, true_crit: str, false_crit: str) -> bool:
        """
        Zwraca szybką decyzję True/False z wykorzystaniem endpointu System One.
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
            # Fallback dla standardowego interfejsu ollama z wymuszeniem JSON
            res = ollama.chat(
                model="llama3.2:1b",
                messages=[
                    {"role": "user", "content": f"Oceń stan: {state_text}\nPytanie: {question}"}
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
        
        # Pobieramy uproszczony tekst strony
        content = await page.inner_text("body")
        sample_state = content[:1500]

        print("[System 1] Analizuję stronę pod kątem obecności formularza logowania...")
        is_login = await router.decide_binary(
            state_text=sample_state,
            question="Czy na stronie jest aktualnie wyświetlony widoczny formularz logowania?",
            true_crit="Widoczne są pola 'username' i 'password' do wpisania danych.",
            false_crit="To jest zwykła strona z listą artykułów/treścią."
        )
        print(f"-> Wynik szybkiej decyzji: {is_login}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
