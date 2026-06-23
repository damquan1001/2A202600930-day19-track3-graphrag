import os, json

dataset_dir = r"E:\Development\VinAI\Day19\2A202600930-day19-track3-graphrag\dataset"
stats = []
for fname in sorted(os.listdir(dataset_dir), key=lambda x: int(x.split('_')[1].split('.')[0])):
    fpath = os.path.join(dataset_dir, fname)
    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    stats.append({
        'file': fname,
        'size_chars': len(content),
        'size_lines': content.count('\n'),
        'query': content.split('Query: ')[1].split('\n')[0] if 'Query: ' in content else 'N/A'
    })

# Summary
queries = set(s['query'] for s in stats)
print(f"Total files: {len(stats)}")
print(f"Unique queries: {len(queries)}")
for q in sorted(queries):
    count = sum(1 for s in stats if s['query'] == q)
    print(f"  [{count}x] {q}")
print(f"\nTotal content size: {sum(s['size_chars'] for s in stats):,} chars")
print(f"File sizes: min={min(s['size_chars'] for s in stats):,}, max={max(s['size_chars'] for s in stats):,}, avg={sum(s['size_chars'] for s in stats)/len(stats):,.0f} chars")
