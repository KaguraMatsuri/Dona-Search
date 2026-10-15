#!/usr/bin/env python3
"""Generate query processor training data using API"""
import json, os, sys
from pathlib import Path
from anthropic import Anthropic

OUTPUT = Path("data/query_train.jsonl")
N_SAMPLES = 5000

PROMPT = """Generate a search query and extract its keywords and filters.

Output JSON format:
{
  "query": "natural language search query",
  "keywords": ["key1", "key2"],
  "filters": {"time": "recent"|"any", "type": "article"|"any"}
}

Generate diverse queries across: tech, science, news, how-to, product reviews, academic research.
JSON:"""

def main():
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    OUTPUT.parent.mkdir(exist_ok=True)

    with open(OUTPUT, "w") as f:
        for i in range(N_SAMPLES):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=256,
                    messages=[{"role": "user", "content": PROMPT}]
                )

                text = response.content[0].text
                data = json.loads(text)

                f.write(json.dumps(data, ensure_ascii=False) + "\n")

                if (i + 1) % 100 == 0:
                    print(f"Generated {i + 1}/{N_SAMPLES}")

            except Exception as e:
                print(f"Error at {i}: {e}")
                continue

    print(f"✓ Saved to {OUTPUT}")

if __name__ == "__main__":
    main()
