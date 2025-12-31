import json

with open('data_new/fixed_test_pairs.json', encoding='utf-8') as f:
    data = json.load(f)

cats1 = data['categories1']
cats2 = data['categories2']
labels = data['labels']

cross_pos = sum(1 for c1,c2,l in zip(cats1,cats2,labels) if c1!=c2 and l==1)
cross_neg = sum(1 for c1,c2,l in zip(cats1,cats2,labels) if c1!=c2 and l==0)
same_pos = sum(1 for c1,c2,l in zip(cats1,cats2,labels) if c1==c2 and l==1)
same_neg = sum(1 for c1,c2,l in zip(cats1,cats2,labels) if c1==c2 and l==0)

print(f'Cross-category positives: {cross_pos}')
print(f'Cross-category negatives: {cross_neg}')
print(f'Same-category positives: {same_pos}')
print(f'Same-category negatives: {same_neg}')
print(f'\nTotal positives: {sum(labels)}')
print(f'Total negatives: {len(labels) - sum(labels)}')
