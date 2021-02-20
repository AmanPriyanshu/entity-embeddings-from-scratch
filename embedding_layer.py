import torch
import numpy as np

class EntityEmbedding(torch.nn.Module):
	def __init__(self, vocab_size, hidden_size, input_size):
		super(EntityEmbedding, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.softmax = torch.nn.Softmax(dim=1)

		weights = torch.Tensor(self.vocab_size, self.hidden_size)
		self.weights = torch.nn.Parameter(weights)
		torch.nn.init.kaiming_uniform_(self.weights, a=np.sqrt(5))

	def forward(self, x):
		x_onehot = torch.FloatTensor(x.shape[0], self.input_size, self.vocab_size)
		x_onehot.zero_()
		for i in range(x.shape[0]):
			for row in range(x[i].shape[0]):
				x_onehot[i][row][x[i][row]] += 1
		w_times_x=torch.zeros(x.shape[0], self.input_size, self.hidden_size)
		for i in range(x.shape[0]):
			w_times_x[i] = torch.mm(x_onehot[i], self.weights)
		return torch.mean(w_times_x, 1)

if __name__ == '__main__':
	layer = EntityEmbedding(10, 2, 3)
	print(layer(torch.tensor([[0, 1, 2], [0, 1, 5]])))