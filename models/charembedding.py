import torch
import torch.nn as nn
class charEmbedding(nn.Module):
    def __init__(self, conv_filter_sizes, conv_filter_nums, char2Idx, char_emb_size, device):
        super(charEmbedding, self).__init__()
        self.conv_filter_sizes = conv_filter_sizes
        self.conv_filter_nums = conv_filter_nums
        self.char2Idx = char2Idx
        self.char_emb_size = char_emb_size
        self.device = device

        self.CharEmbedding = nn.Embedding(len(char2Idx), self.char_emb_size, padding_idx=char2Idx['<PAD>'])
        self.char_encoders = []

        for i, filter_size in enumerate(self.conv_filter_sizes):
            f = nn.Conv3d(in_channels=1, out_channels=self.conv_filter_nums[i],
                          kernel_size=(1, filter_size, self.char_emb_size))
            self.char_encoders.append(f)
        for conv in self.char_encoders:
            if self.device != 'cpu':
                conv.cuda(self.device)

    def forward(self, inputs):
        """
        Args:
            inputs: 3D tensor, [bs, max_len, max_len_char]
        Returns:
            char_conv_outputs: 3D tensor, [bs, max_len, output_dim]
        """
        max_len, max_len_char = inputs.size(1), inputs.size(2)
        inputs = inputs.view(-1, max_len * max_len_char)  # [bs, -1]
        input_embed = self.CharEmbedding(inputs)  # [bs, ml*ml_c, feature_dim]
        # input_embed = self.dropout_embed(input_embed)
        # [bs, 1, max_len, max_len_char, feature_dim]
        input_embed = input_embed.view(-1, 1, max_len, max_len_char, self.char_emb_size)
        # conv
        char_conv_outputs = []
        for char_encoder in self.char_encoders:
            conv_output = char_encoder(input_embed)
            pool_output = torch.squeeze(torch.max(conv_output, -2)[0], -1)
            char_conv_outputs.append(pool_output)
        char_conv_outputs = torch.cat(char_conv_outputs, dim=1)
        char_conv_outputs = char_conv_outputs.permute(0, 2, 1)

        return char_conv_outputs