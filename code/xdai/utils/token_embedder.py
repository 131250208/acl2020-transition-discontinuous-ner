import inspect, os, torch
import numpy as np
from typing import Dict
from xdai.utils.nn import TimeDistributed
from xdai.utils.seq2vec import CnnEncoder
from xdai.elmo.models import Elmo
from IPython.core.debugger import set_trace
from transformers import BertModel
from transformers import BertTokenizerFast

'''Update date: 2019-Nov-5'''
class Embedding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, weight: torch.FloatTensor = None, trainable=True):
        super(Embedding, self).__init__()
        self.output_dim = embedding_dim

        if weight is None:
            weight = torch.FloatTensor(vocab_size, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)
            torch.nn.init.xavier_uniform_(self.weight)
        else:
            assert weight.size() == (vocab_size, embedding_dim)
            self.weight = torch.nn.Parameter(weight, requires_grad=trainable)


    def get_output_dim(self):
        return self.output_dim


    def forward(self, inputs):
        outs = torch.nn.functional.embedding(inputs, self.weight)
        return outs


'''Update date: 2019-Nov-5'''
class TokenCharactersEmbedder(torch.nn.Module):
    def __init__(self, embedding: Embedding, encoder, dropout=0.0):
        super(TokenCharactersEmbedder, self).__init__()
        self._embedding = TimeDistributed(embedding)
        self._encoder = TimeDistributed(encoder)
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x


    def get_output_dim(self):
        return self._encoder._module.get_output_dim()


    def forward(self, token_characters):
        '''token_characters: batch_size, num_tokens, num_characters'''
        mask = (token_characters != 0).long()
        outs = self._embedding(token_characters)
        outs = self._encoder(outs, mask)
        outs = self._dropout(outs)
        return outs


class MyBert(torch.nn.Module):
    def __init__(self, tokenizer, bert):
        super(MyBert, self).__init__()
        self.tokenizer = tokenizer
        self.bert = bert

    def forward(self, *args, **kwargs):
        return self.bert(*args, **kwargs)

    def get_output_dim(self):
        return self.bert.config.hidden_size


'''A single layer of ELMo representations, essentially a wrapper around ELMo(num_output_representations=1, ...)
Update date: 2019-Nov-5'''
class ElmoTokenEmbedder(torch.nn.Module):
    def __init__(self, options_file, weight_file, dropout=0.5, requires_grad=False,
                 projection_dim=None):
        super(ElmoTokenEmbedder, self).__init__()

        self._elmo = Elmo(options_file, weight_file, num_output_representations=1, dropout=dropout,
                          requires_grad=requires_grad)
        if projection_dim:
            self._projection = torch.nn.Linear(self._elmo.get_output_dim(), projection_dim)
            self.output_dim = projection_dim
        else:
            self._projection = None
            self.output_dim = self._elmo.get_output_dim()


    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs):
        # inputs: batch_size, num_tokens, 50
        elmo_output = self._elmo(inputs)
        elmo_representations = elmo_output["elmo_representations"][0]
        if self._projection:
            projection = self._projection
            for _ in range(elmo_representations.dim() - 2):
                projection = TimeDistributed(projection)
            elmo_representations = projection(elmo_representations)
        return elmo_representations


'''Update date: 2019-Nov-5'''
def _load_pretrained_embeddings(filepath, dimension, token2idx):
    tokens_to_keep = set(token2idx.keys())
    embeddings = {}
    if filepath != "" and os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                sp = line.strip().split(" ")
                if len(sp) <= dimension: continue
                token = sp[0]
                if token not in tokens_to_keep: continue
                embeddings[token] = np.array([float(x) for x in sp[1:]])

    print(" # Load %d out of %d words (%d-dimensional) from pretrained embedding file (%s)!" % (
    len(embeddings), len(token2idx), dimension, filepath))

    all_embeddings = np.asarray(list(embeddings.values()))
    embeddings_mean = float(np.mean(all_embeddings))
    embeddings_std = float(np.std(all_embeddings))

    weights = np.random.normal(embeddings_mean, embeddings_std, size=(len(token2idx), dimension))
    for token, i in token2idx.items():
        if token in embeddings:
            weights[i] = embeddings[token]
    return weights


'''Reference url: https://github.com/allenai/allennlp/blob/master/allennlp/modules/text_field_embedders/*
Takes as input the dict produced by TextField and 
returns as output an embedded representations of the tokens in that field
Update date: 2019-Nov-5'''
class TextFieldEmbedder(torch.nn.Module):
    def __init__(self, token_embedders, embedder_to_indexer_map=None, vocab=None):
        super(TextFieldEmbedder, self).__init__()
        self.vocab = vocab
        self.token_embedders = token_embedders
        self._embedder_to_indexer_map = embedder_to_indexer_map
        for k, embedder in token_embedders.items():
            if k == "bert":
                # self.bert_vocab = embedder.tokenizer.get_vocab()
                self.bert_tokenizer = embedder.tokenizer
            self.add_module("token_embedder_%s" % k, embedder)

    def get_output_dim(self):
        return sum([embedder.get_output_dim() for embedder in self.token_embedders.values()])


    '''text_field_input is the output of a call to TextField.as_tensor (see instance.py).
    Each tensor in here is assumed to have a shape roughly similar to (batch_size, num_tokens)'''
    def forward(self, text_field_input: Dict[str, torch.Tensor], **kwargs):
        outs = []

        if "bert" in self.token_embedders:
            embedder = getattr(self, "token_embedder_%s" % "bert")
            forward_params = inspect.signature(embedder.forward).parameters
            forward_params_values = {}
            for param in forward_params.keys():
                if param in kwargs:
                    forward_params_values[param] = kwargs[param]

            # (batch_size, seq_len)
            tok_ids = text_field_input["tokens"]
            device = tok_ids.device
            batch_bert_inp_ids = []

            new_word_ids = []
            new_token_characters = []
            for l_idx, l in enumerate(tok_ids):
                words = [self.vocab.get_item_from_index(t.item()) for t in l]
                bert_toks = []
                subwd2wd = []
                for idx, w in enumerate(words):
                    if w == "@@PADDING@@":
                        bert_toks.append("[PAD]")
                        subwd2wd.append(idx)
                    elif w == "@@UNKNOWN@@":
                        bert_toks.append("[UNK]")
                        subwd2wd.append(idx)
                    else:
                        tks = self.bert_tokenizer.tokenize(w)
                        bert_toks.extend(tks)
                        subwd2wd.extend([idx] * len(tks))
                bert_ids = [self.bert_tokenizer.get_vocab()[t] for t in bert_toks]
                batch_bert_inp_ids.append(bert_ids)
                assert len(bert_ids) == len(subwd2wd)
                subwd2wd = torch.LongTensor(subwd2wd).to(device)
                new_word_ids.append(torch.index_select(text_field_input["tokens"][l_idx], 0, subwd2wd))
                new_token_characters.append(
                    torch.index_select(text_field_input["token_characters"][l_idx], 0, subwd2wd))
            text_field_input["tokens"] = torch.stack(new_word_ids, dim=0).to(device)
            text_field_input["token_characters"] = torch.stack(new_token_characters, dim=0).to(device)
            batch_bert_inp_ids = torch.LongTensor(batch_bert_inp_ids).to(device)
            mask = torch.ones_like(batch_bert_inp_ids).to(device)
            tok_type = torch.zeros_like(batch_bert_inp_ids).to(device)
            set_trace()
            outs.append(embedder(batch_bert_inp_ids, mask, tok_type, **forward_params_values)[0])

        for k in sorted(self.token_embedders.keys()):
            if k == "bert":
                continue
            embedder = getattr(self, "token_embedder_%s" % k)
            forward_params = inspect.signature(embedder.forward).parameters
            forward_params_values = {}
            for param in forward_params.keys():
                if param in kwargs:
                    forward_params_values[param] = kwargs[param]
            if self._embedder_to_indexer_map is not None and k in self._embedder_to_indexer_map:
                indexer_map = self._embedder_to_indexer_map[k]
                assert isinstance(indexer_map, dict)
                tensors = {name: text_field_input[argument] for name, argument in indexer_map.items()}
                outs.append(embedder(**tensors, **forward_params_values))
            else:
                tensors = [text_field_input[k]]
                outs.append(embedder(*tensors, **forward_params_values))
        return torch.cat(outs, dim=-1)


    @classmethod
    def tokens_embedder(cls, vocab, args):
        token2idx = vocab.get_item_to_index_vocabulary("tokens")
        weight = _load_pretrained_embeddings(args.pretrained_word_embeddings, dimension=100, token2idx=token2idx)
        return Embedding(len(token2idx), embedding_dim=100, weight=torch.FloatTensor(weight))


    @classmethod
    def token_characters_embedder(cls, vocab, args):
        embedding = Embedding(vocab.get_vocab_size("token_characters"), embedding_dim=16)
        return TokenCharactersEmbedder(embedding, CnnEncoder())


    @classmethod
    def elmo_embedder(cls, vocab, args):
        option_file = os.path.join(args.pretrained_model_dir, "options.json")
        weight_file = os.path.join(args.pretrained_model_dir, "weights.hdf5")
        return ElmoTokenEmbedder(option_file, weight_file)


    @classmethod
    def create_embedder(cls, args, vocab):
        embedder_to_indexer_map = {}
        embedders = {"tokens": TextFieldEmbedder.tokens_embedder(vocab, args),
                 "token_characters": TextFieldEmbedder.token_characters_embedder(vocab, args)}

        if args.model_type == "elmo":
            embedders["elmo_characters"] = TextFieldEmbedder.elmo_embedder(vocab, args)

        if args.model_type == "bert":
            bert_path = args.pretrained_model_dir
            embedders["bert"] = MyBert(BertTokenizerFast.from_pretrained(bert_path),
                                       BertModel.from_pretrained(bert_path))

        return cls(embedders, embedder_to_indexer_map, vocab)