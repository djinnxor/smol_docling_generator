from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "ibm-granite/granite-docling-258M"
)

text = "Монгол Улсын Их Сургууль"

enc = tokenizer(text, add_special_tokens=False)

print(enc.input_ids)
print(len(enc.input_ids))

for tid in enc.input_ids:
    print(tid, tokenizer.decode([tid]))