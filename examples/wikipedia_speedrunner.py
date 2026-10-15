"""
Wikipedia Semantic Speedrunner (System 1 na GPU RTX 4070 Ti).
Wyzwanie: Przejście z artykułu 'Most Grunwaldzki' do 'Clang' (kompilator C/C++)
Poprzez ocenę dziesiątek linków na żywo na GPU w kilkanaście milisekund!
"""

import asyncio
import time
import urllib.parse
import torch
from typing import List, Dict, Any
from playwright.async_api import async_playwright
from transformers import AutoTokenizer, AutoModel


class SemanticNavigator:
    def __init__(self, model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device: str = "cuda"):
        print(f"[SemanticNavigator] Inicjalizacja modelu {model_id} na GPU ({device})...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()

    def encode(self, texts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            emb = out.last_hidden_state.mean(dim=1)
            return torch.nn.functional.normalize(emb, p=2, dim=1)

    def rank_links(self, target_description: str, candidate_links: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        if not candidate_links:
            return []

        t_emb = self.encode([target_description])
        cand_texts = [f"{c['text']} - {c['clean_slug']}" for c in candidate_links]
        c_emb = self.encode(cand_texts)

        sims = (c_emb @ t_emb.T).squeeze(-1).tolist()
        if not isinstance(sims, list):
            sims = [sims]

        ranked = []
        for c, s in zip(candidate_links, sims):
            score = float(s)
            clean_name = c['clean_slug'].lower()
            # Heurystyczny bonus, gdy pojawia się kluczowe pojęcie
            if "c++" in clean_name or "c++" in c['text'].lower():
                score += 0.15
            elif "kompilator" in clean_name or "compiler" in clean_name:
                score += 0.20
            elif "llvm" in clean_name:
                score += 0.30

            ranked.append({**c, "score": round(score, 4), "raw_sim": round(float(s), 4)})

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked


async def run_wikipedia_challenge():
    print("=" * 80)
    print("🚀 WIKIPEDIA SPEEDRUN: MOST GRUNWALDZKI ➡️ KOMPILATOR CLANG")
    print("=" * 80)

    start_url = "https://pl.wikipedia.org/wiki/Most_Grunwaldzki_we_Wroc%C5%82awiu"
    target_keyword = "clang"
    target_semantic = (
        "kompilator Clang dla języka C i C++ narzędzie programistyczne LLVM kod maszynowy"
    )

    navigator = SemanticNavigator()
    visited_urls = set()
    history_path = []
    t_speedrun_start = time.perf_counter()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
        )

        current_url = start_url
        print(f"\n[START] Wchodzę na stronę początkową: {current_url}")
        await page.goto(current_url, wait_until="domcontentloaded")

        max_hops = 12
        for hop in range(1, max_hops + 1):
            curr_title = await page.title()
            curr_url = page.url
            visited_urls.add(curr_url)
            clean_title = curr_title.split("–")[0].strip()
            history_path.append(clean_title)

            print(f"\n--- SKOK #{hop} ---")
            print(f"Bieżący artykuł: \"{clean_title}\"")
            print(f"URL: {curr_url}")

            # Sprawdzenie warunku sukcesu
            unquoted_url = urllib.parse.unquote(curr_url).lower()
            if target_keyword in unquoted_url or target_keyword in curr_title.lower():
                total_duration = time.perf_counter() - t_speedrun_start
                print("\n" + "🎉" * 40)
                print(f"🎯 CEL OSIĄGNIĘTY W {hop-1} SKOKACH! (Czas łączny: {total_duration:.2f} s)")
                print(f"Dotarliśmy do: {curr_title}")
                print("Ścieżka przejścia:")
                for i, step in enumerate(history_path, 1):
                    print(f"  {i}. {step}")
                print("🎉" * 40)
                break

            # 1. Ekstrakcja linków
            raw_links = await page.evaluate("""() => {
                const content = document.querySelector('#mw-content-text');
                if (!content) return [];
                const anchors = Array.from(content.querySelectorAll('p a, li a, table.infobox a'));
                return anchors.map(a => {
                    const text = a.innerText.trim();
                    const href = a.href;
                    return { text, href };
                }).filter(x => {
                    if (!x.href.includes('pl.wikipedia.org/wiki/')) return false;
                    if (x.href.includes('redlink=1') || x.href.includes('action=edit')) return false;
                    const slug = x.href.split('/wiki/')[1] || '';
                    if (slug.includes(':') || slug.includes('#')) return false;
                    return x.text.length > 1;
                });
            }""")

            candidates = []
            seen_slugs = set()
            for l in raw_links:
                slug = l["href"].split("/wiki/")[1].split("?")[0]
                clean_slug = urllib.parse.unquote(slug).replace("_", " ")
                if slug not in seen_slugs and l["href"] not in visited_urls:
                    seen_slugs.add(slug)
                    candidates.append({
                        "text": l["text"],
                        "href": l["href"],
                        "slug": slug,
                        "clean_slug": clean_slug
                    })

            print(f"Liczba kandydatów merytorycznych: {len(candidates)}")

            # Sprawdzenie bezpośredniego strzału w Clang
            direct_hit = next((c for c in candidates if target_keyword in c["clean_slug"].lower() or target_keyword in c["text"].lower()), None)
            if direct_hit:
                print(f"  🔥 Bezpośredni strzał w cel wykryty: '{direct_hit['text']}' -> {direct_hit['href']}!")
                await page.goto(direct_hit["href"], wait_until="domcontentloaded")
                continue

            # 2. Ranking semantyczny na GPU
            t0 = time.perf_counter()
            ranked = navigator.rank_links(target_semantic, candidates)
            latency_ms = (time.perf_counter() - t0) * 1000

            print(f"Czas ewaluacji na RTX 4070 Ti: {latency_ms:.1f} ms")
            print("Top 3 kierunki:")
            for rank_i, c in enumerate(ranked[:3], 1):
                print(f"  {rank_i}. [{c['score']:.4f}] '{c['text']}' -> {c['clean_slug']}")

            best = ranked[0]
            print(f"➡️ Wybrano: '{best['text']}' -> {best['clean_slug']}")
            await page.goto(best["href"], wait_until="domcontentloaded")
            await page.wait_for_timeout(400)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_wikipedia_challenge())
