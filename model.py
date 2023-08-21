"""
Author: ByronVon
Date: 2023-08-17 16:06:19
FilePath: /exercises/models/seq2seq/model.py
Description: bert作为encoder,接一个单向的gru解码，将口语化的日期解析为YYYYMMDD
"""


import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super(Attention, self).__init__()
        self.fc = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, dec_hidden_dim]
        # encoder_outputs: [batch_size, seq_len, enc_hidden_dim]

        # Repeat decoder hidden state across all times steps of encoder outputs
        seq_len = encoder_outputs.shape[1]

        # [batch_size, dec_hidden_dim]->[batch_size, 1, dec_hidden_size]->[batch_size, seq_len, dec_hidden_size]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # Calculate attention energies
        energy = torch.tanh(self.fc(torch.cat((hidden, encoder_outputs), dim=2)))

        # Calculate attention weights
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]

        # Apply Softmax
        scores = self.softmax(attention)  # [batch_size, seq_len]
        return scores


class DateConversionModel(nn.Module):
    # 将BERT的last_hidden_state最后一个时间步的输出（即句子的最后一个token的输出）用作解码器的初始隐藏状态。
    # 初始输入到解码器是一个特殊的开始token或[CLS]的嵌入。
    # 使用循环来连续地产生8个数字[YYYYMMDD]
    def __init__(self, encoder_model_name, num_classes, target_len, cls_token_id) -> None:
        super().__init__()
        # bert encoder
        self.encoder = AutoModel.from_pretrained(encoder_model_name)

        # gru layers
        # (batch_size, seq_len, hidden_size)->(batch_size, seq_len, gru_hidden)
        self.decoder = nn.GRU(
            input_size=self.encoder.config.hidden_size,
            hidden_size=self.encoder.config.hidden_size,
            bidirectional=False,
            batch_first=True,
        )
        # (batch_size, seq_len, gru_hidden)->(batch_size, seq_len, num_classes)
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)
        self.len = target_len
        self.cls_token_id = cls_token_id  # 从cls_token作为开始

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # pass the input through BERT
        # last_hidden_state: [batch_size, seq_len, hidden_size]，表示最后一层隐藏层的输出，一般用于序列标注和文本生成
        # pooler_output: [batch_size, hidden_size]，表示[cls]，一般用于分类
        encoder_output = self.encoder(input_ids, attention_mask)[0]  # last_hidden_state
        hidden = encoder_output[:, -1, :].unsqueeze(0)  # use last output of BERT as initialial hidden state for GRU

        # Starting input to the GRU decoder - using embedding of [CLS]
        input_token = torch.full((input_ids.size(0), 1), self.cls_token_id, dtype=torch.long).to(input_ids.device)
        decoder_input = self.encoder.embeddings(input_token)

        # Decoding loop to get N outputs
        outputs = []
        for _ in range(self.len):
            out, hidden = self.decoder(decoder_input, hidden)
            decoder_input = out
            out = self.fc(out.squeeze(1))
            outputs.append(out)

        return torch.stack(tensors=outputs, dim=1)  # shape: [batch_size, 8, 10]


class DateConversionModelv2(nn.Module):
    def __init__(self, encoder_model_name, num_classes, target_len, cls_token_id, dropout_ratio=0.2) -> None:
        super().__init__()

        # bert encoder
        self.encoder = AutoModel.from_pretrained(encoder_model_name)

        # gru layers
        self.decoder = nn.GRU(
            input_size=self.encoder.config.hidden_size,
            hidden_size=self.encoder.config.hidden_size,
            bidirectional=False,
            batch_first=True,
        )

        # Linear layer to get the final predicted class
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)

        # Fixed length of the target sequence
        self.len = target_len
        self.cls_token_id = cls_token_id

        # Dropout
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input_ids, attention_mask, token_type_ids=None, targets=None, teacher_forcing_ratio=0.5):
        # BERT encoder
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        encoder_output = self.dropout(encoder_output)
        # Using the last output of BERT as initial hidden state for GRU
        hidden = encoder_output[:, -1, :].unsqueeze(0).contiguous()

        # Starting input to the GRU decoder using embedding of [CLS]
        decoder_input = torch.full((input_ids.size(0), 1), self.cls_token_id, dtype=torch.long).to(input_ids.device)
        decoder_input = self.encoder.embeddings(decoder_input)

        # Decoding loop
        outputs = []
        for t in range(self.len):
            out, hidden = self.decoder(decoder_input, hidden)

            # Apply the linear layer to get the output class
            out = self.fc(out.squeeze(1))  # 去除max_len维度
            outputs.append(out)

            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if targets is not None and use_teacher_forcing:
                next_input = targets[:, t].unsqueeze(1)
                decoder_input = self.encoder.embeddings(next_input)
            else:
                # Use the top predicted token
                top1 = out.argmax(1).unsqueeze(1)
                decoder_input = self.encoder.embeddings(top1)

        return torch.stack(outputs, dim=1)  # [batch_size, len, num_classes]


class DateConversionModelv3(nn.Module):
    """在decoder的每一步加入attention"""

    def __init__(self, encoder_model_name, num_classes, target_len, cls_token_id, dropout_ratio=0.2) -> None:
        super().__init__()

        # bert encoder
        self.encoder = AutoModel.from_pretrained(encoder_model_name)

        # attention
        self.attention = Attention(self.encoder.config.hidden_size, self.encoder.config.hidden_size)

        # gru decoder
        self.decoder = nn.GRU(
            input_size=self.encoder.config.hidden_size * 2,
            hidden_size=self.encoder.config.hidden_size,
            bidirectional=False,
            batch_first=True,
        )

        # Linear layer to get the final predicted class
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)

        # Fixed length of the target sequence
        self.len = target_len
        self.cls_token_id = cls_token_id

        # Dropout
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, input_ids, attention_mask, token_type_ids=None, targets=None, teacher_forcing_ratio=0.5):
        # BERT encoder
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        encoder_output = self.dropout(encoder_output)
        # Using the last output of BERT as initial hidden state for GRU
        hidden = encoder_output[:, -1, :].unsqueeze(0).contiguous()

        # Starting input to the GRU decoder using embedding of [CLS]
        decoder_input = torch.full((input_ids.size(0), 1), self.cls_token_id, dtype=torch.long).to(input_ids.device)
        decoder_input = self.encoder.embeddings(decoder_input)

        # Decoding loop
        outputs = []
        for t in range(self.len):
            # Calculate attention score with hidden
            scores = self.attention(hidden.squeeze(0), encoder_output)  # [batch_size, seq_len]
            context = torch.bmm(scores.unsqueeze(1), encoder_output)  # [batch_size, 1, seq_len]
            # Concat context with decoder input
            gru_input = torch.cat((decoder_input, context), dim=2)  # [batch_size, 1, enc_hidden_dim*2]
            # 修改这里
            out, hidden = self.decoder(gru_input, hidden)

            # Apply the linear layer to get the output class
            out = self.fc(out.squeeze(1))  # 去除max_len维度
            outputs.append(out)

            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if targets is not None and use_teacher_forcing:
                next_input = targets[:, t].unsqueeze(1)
                decoder_input = self.encoder.embeddings(next_input)
            else:
                # Use the top predicted token
                top1 = out.argmax(1).unsqueeze(1)
                decoder_input = self.encoder.embeddings(top1)

        return torch.stack(outputs, dim=1)  # [batch_size, len, num_classes]


def test_attention():
    # Define some parameters
    batch_size = 5
    seq_len = 10
    enc_hidden_dim = 256
    dec_hidden_dim = 256

    # Create random tensors
    hidden = torch.randn(batch_size, dec_hidden_dim)
    encoder_outputs = torch.randn(batch_size, seq_len, enc_hidden_dim)

    # Initialize attention module
    attention = Attention(enc_hidden_dim, dec_hidden_dim)

    # Calculate attention weights
    attn_weights = attention(hidden, encoder_outputs)

    # Assert attention weights size
    assert attn_weights.size() == (
        batch_size,
        seq_len,
    ), f"Expected size {(batch_size, seq_len)}, but got {attn_weights.size()}"

    # Assert attention weights sum up to 1
    assert torch.allclose(
        attn_weights.sum(dim=1), torch.tensor([1.0] * batch_size, dtype=torch.float32), atol=1e-5
    ), f"Attention weights do not sum up to 1."

    print("All tests passed!")


if __name__ == "__main__":
    # test_attention()

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DateConversionModelv3(model_name, 10, 8, tokenizer.cls_token_id)  # len(positions)*num_classes

    text = ["march ten", "twenty and twenty three"]
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print(inputs)
    output = model(**inputs)
    print(output.size())
    print(model)
