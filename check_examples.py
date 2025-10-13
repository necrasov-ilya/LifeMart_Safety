import pandas as pd
from pathlib import Path

df = pd.read_csv('data/messages.csv')
spam_df = df[df['label'] == 1]
spam_df['length'] = spam_df['message'].str.len()

filtered = spam_df[(spam_df['length'] >= 50) & (spam_df['length'] <= 300)]

examples = filtered.sample(n=12, random_state=42)

print('Selected 12 examples with lengths:')
print('=' * 80)
for idx, row in examples.iterrows():
    print(f'{row["length"]:3d} chars: {row["message"][:80]}')
    if len(row["message"]) > 80:
        print(f'           ...{row["message"][80:160]}')
    print()
