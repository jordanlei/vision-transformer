import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc1(x)
        return x

class SingleHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SingleHeadAttention, self).__init__() 
        self.hidden_size = hidden_size
        self.W_q = nn.Linear(input_size, self.hidden_size, bias = False)
        self.W_k = nn.Linear(input_size, self.hidden_size, bias = False)
        self.W_v = nn.Linear(input_size, self.hidden_size, bias = False)
    
    def forward(self, x): 
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        self.attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(self.attention_weights, v)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = num_heads * hidden_size
        self.attention_heads = nn.ModuleList([SingleHeadAttention(input_size, hidden_size) for _ in range(num_heads)])
        self.W_o = nn.Linear(self.dim, output_size)

    def forward(self, x):
        attention_outputs = [head(x) for head in self.attention_heads]
        x = torch.cat(attention_outputs, dim=-1)
        x = self.W_o(x)
        return x
        
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(TransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, hidden_size, hidden_size, num_heads)
        self.net = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size))
        
    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.net(self.norm2(x))
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, channels, img_size=32, patch_size=4, hidden_size=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        # Each patch is patch_size x patch_size pixels
        # Total number of patches is (img_size/patch_size)^2
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(channels, hidden_size, kernel_size=patch_size, stride=patch_size) 
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        # Positional embedding for patches + CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size))
        
    def forward(self, x):
        # x: [B, C, H, W] -> [B, num_patches, hidden_size]
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        
        # prepend CLS token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # add positional embeddings
        x = x + self.pos_embed
        return x

class VisionTransformer(nn.Module): 
    def __init__(self, img_size, hidden_size, output_size, num_heads, num_blocks):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(3, img_size, patch_size = 4, hidden_size = hidden_size)
        self.blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads) for _ in range(num_blocks)])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        for block in self.blocks:
            x =  block(x)   
        x = x[:, 0]
        x = self.fc(x)
        return x

