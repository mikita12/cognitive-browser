import asyncio
import time

import torch
from playwright.async_api import async_playwright
from transformers import AutoModelForSequenceClassification, AutoTokenizer


async def main():
    print("=" * 60)
    print("TEST: ModernBERT (8192 context) vs Całe Drzewo Dostępności")
    print("=" * 60)

    # 1. Pobranie całego drzewa ARIA ze strony głównej Wikipedii
    print("\n1. Otwieram przeglądarkę i pobieram CAŁE Accessibility Tree...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        t0 = time.perf_counter()
        await page.goto("https://en.wikipedia.org/wiki/Main_Page", timeout=15000)
        full_tree = await page.aria_snapshot()
        fetch_ms = (time.perf_counter() - t0) * 1000
        await browser.close()

    print(f"-> Pobrane drzewo ma {len(full_tree)} znaków i {len(full_tree.splitlines())} linii.")
    print(f"-> Czas pobrania strony + zrzutu drzewa: {fetch_ms:.1f} ms")

    # 2. Tokenizacja całym drzewem w ModernBERT
    model_name = "Alibaba-NLP/gte-reranker-modernbert-base"
    print(f"\n2. Ładowanie tokenizera i modelu {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokens = tokenizer.encode(full_tree)
    token_count = len(tokens)
    print(f"-> LICZBA TOKENÓW CAŁEGO DRZEWA WIKIPEDII: {token_count} tokenów!")
    print(f"-> Maksymalny limit ModernBERT: {tokenizer.model_max_length} tokenów.")
    if token_count <= tokenizer.model_max_length:
        print(f"-> SUKCES: Całe drzewo strony mieści się w ModernBERT w 100% (zajmuje {(token_count/8192)*100:.1f}% bufora)!")
    else:
        print(f"-> Drzewo wymaga lekkiego przycięcia ({token_count} > {tokenizer.model_max_length}).")

    # 3. Ładowanie modelu na GPU RTX 4070 Ti
    print("\n3. Ładowanie wag modelu do pamięci GPU (RTX 4070 Ti, bfloat16)...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to("cuda")
    model.eval()

    # 4. Sprawdzenie inferencji z CAŁYM drzewem jako kontekstem
    intents = [
        "Wyszukaj artykuł w polu wyszukiwania",
        "Przejdź do portalu Current events",
        "Zaloguj się na konto Wikipedii (Log in)",
        "Sprawdź dzisiejszy wyróżniony artykuł (Today's featured article)",
    ]

    print("\n4. Przepuszczam CAŁE drzewo przez model dla różnych intencji...")
    for intent in intents:
        # Wejście: (Intencja, Całe drzewo)
        inputs = tokenizer(
            [[intent, full_tree]],
            padding=True,
            truncation=True,
            max_length=8192,
            return_tensors="pt"
        ).to("cuda")

        torch.cuda.synchronize()
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits.squeeze()
        torch.cuda.synchronize()
        latency = (time.perf_counter() - t_start) * 1000

        allocated_vram = torch.cuda.memory_allocated() / (1024 ** 2)
        print(f"Intencja: '{intent}'")
        print(f"   Score: {logits.item():.4f} | Czas inferencji: {latency:.2f} ms | VRAM: {allocated_vram:.0f} MB")

    print("\n" + "=" * 60)
    print("WNIOSKI I PODSUMOWANIE:")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
