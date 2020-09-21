from transformers import BertTokenizer, BertModel, GPT2Model, BertForMultipleChoice

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
#model.cuda()

print(tokenizer.encode('Minecraft is a sandbox video game developed by Mojang', add_special_tokens=False))
print(tokenizer.encode('Minecraft', add_special_tokens=False))
print(tokenizer.encode('sandbox', add_special_tokens=False))