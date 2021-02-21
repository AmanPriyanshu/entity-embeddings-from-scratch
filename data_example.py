import torch
from embedding_layer import EntityEmbedding

data = ['cat chases mice',
		'cat catches mice',
		'cat eats mice',
		'mice runs into hole',
		'cat says bad words',
		'cat and mice are pals',
		'cat and mice are chums',
		'mice stores food in hole',
		'cat stores food in house',
		'mice sleeps in hole',
		'cat sleeps in house',
		'cat and mice are buddies',
		'mice lives in hole',
		'cat lives in house bruh ur mum']

data = [i.lower() for i in data]
vocab = list(set(' '.join(data).split()))

word_key = {word:i+1 for i,word in enumerate(vocab)}

def converter(s):
	return [word_key[word] for word in s.split()]

def pad_sequences(data, max_len=10):
	max_len = max_len if max([len(i) for i in data])>max_len else max([len(i) for i in data])
	return [i+[0]*(max_len-len(i)) if len(i)<max_len else i[:max_len] for i in data]

class SimpleModel(torch.nn.Module):
	def __init__(self, vocab_size, hidden_size, input_size):
		super().__init__()
		self.embedding_layer = EntityEmbedding(vocab_size, hidden_size, input_size)
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size

	def forward(self, x):
		x = self.embedding_layer(x)
		x = torch.nn.Linear(self.hidden_size, self.vocab_size)(x)
		x = torch.nn.Softmax(dim=1)(x)
		return x

def generate_dataset(data, n=2):
	x, y = [], []
	for row in data:
		for index in range(0, len(row)-n-1):
			x.append(row[index:index+n])
			y.append(row[index+n])
	return torch.Tensor(x).int(), torch.Tensor(y).long()

def train():
	data_one_hot = [converter(i) for i in data]
	data_padded = pad_sequences(data_one_hot)
	x, y = generate_dataset(data_padded, 1)
	model = SimpleModel(vocab_size=len(vocab)+1, hidden_size=len(vocab)//4, input_size=1)

	criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())

	for epoch in range(500):
		optimizer.zero_grad()
		output = model(x)
		loss = criterion(output, y)
		print(loss)
		loss.backward()
		optimizer.step()

if __name__ == '__main__':
	train()
	