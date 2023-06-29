############################
# 모듈 import
############################
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from torchtext.vocab import build_vocab_from_iterator
from sklearn.metrics import accuracy_score

from tqdm import tqdm

############################
# 사전 클래스 정의
############################
class GerEngVocab():
    def __init__(self, datas):
        
        data_iter = self.__iter_vocab(datas)
        
        sep_list = ['<unk>', '<pad>', '<sos>', '<eos>']
        
        self.vocab = build_vocab_from_iterator(
            data_iter,
            specials=sep_list
        )
        
        self.vocab.set_default_index(self.vocab['<unk>'])
        
        self.vocab_size = len(self.vocab)

        self.idx2word = self.vocab.get_itos()
        self.word2idx = self.vocab.get_stoi()
    
    def __len__(self):
        return self.vocab_size
    
    def __getitem__(self, idx):
        return self.idx2word[idx]
    
    # 단어 -> 인덱스
    def wrd2idx(self, word):
        return self.word2idx[word]
    
    # 인덱스 -> 단어
    def idx2wrd(self, idx):
        return self.idx2word[idx]
    
    # vocab 생성을 위한 iterator
    def __iter_vocab(self, datas):
        for data in tqdm(datas):
            yield data
            
############################
# 데이터셋 클래스 정의
############################
class GerEngDataset(Dataset):
    ########################
    # 필수 함수 정의
    ########################
    def __init__(self, X, y=[], max_len=10, emb_type='ohe', train_mode=True):
        
        self.__train_mode = train_mode if y == [] else False
        self.__emb_type = emb_type
        self.__max_len = max_len
        self.__vocab = GerEngVocab(X+y)
        self.__oh_vector = torch.eye(len(self.__vocab))
        self.__X = self.__tokenizer(X)
        
        if y != []:
            _y = y
            self.__y = self.__tokenizer(y)
            self.__dec_input = self.__add_sep_token(self.__y)
            self.__target = self.__add_sep_token(self.__y, sep_token='<eos>')
            self.__fake_dec_input = self.__add_sep_token(self.__get_initial_dec_input())
        else:
            self.__y = None
            self.__fake_dec_input = self.__add_sep_token(self.__get_initial_dec_input())
            pass
        
    def __len__(self):
        return len(self.__X)
    
    def __getitem__(self, idx):
        if self.__emb_type == 'ohe':
            return self.__ohe_getitem(idx)
        elif self.__emb_type == 'emb':
            return self.__emb_getitem(idx)
        
    ########################
    # 기타 함수 정의
    ########################
    def __ohe_getitem(self, idx):
        ohe_X = self.__oh_encoding(self.__X[idx])
        if self.__train_mode and self.__y is not None:
            ohe_dec_input = self.__oh_encoding(self.__dec_input[idx])
            ohe_target = self.__oh_encoding(self.__target[idx])
            return torch.tensor(ohe_X), torch.tensor(ohe_dec_input), torch.tensor(ohe_target)
        else:
            ohe_dec_input = self.__oh_encoding(self.__fake_dec_input[idx])
            return torch.tensor(ohe_X), torch.tensor(ohe_dec_input), torch.tensor(ohe_dec_input)
    
    def __emb_getitem(self, idx):
        if self.__train_mode and self.__y is not None:
            return self.__X[idx], self.__dec_input[idx], self.__target[idx]
        else:
            return self.__X[idx], self.__fake_dec_input[idx], self.__fake_dec_input[idx].copy()
    
    def __tokenizer(self, datas):
        padding_datas = self.__padding(datas)
        token_datas = []
        for data in padding_datas:
            data = [self.__vocab.wrd2idx(wrd) for wrd in data]
            token_datas.append(data)
        return np.array(token_datas)
    
    def __padding(self, datas):
        padding_datas = []
        for data in datas:
            data = data + ['<pad>' for _ in range(self.__max_len-len(data))]
            padding_datas.append(data)
        return padding_datas
    
    def __oh_encoding(self, datas):
        return self.__oh_vector[self.__vocab.wrd2idx(datas)]
    
    def __add_sep_token(self, token_datas, sep_token='<sos>'):
        added_array = None
        _array = np.array([self.__vocab.wrd2idx(sep_token) for _ in range(len(self.__X))])
        if sep_token == '<sos>':
            added_array = np.concatenate((_array.reshape(-1, 1), token_datas), axis=1)
        elif sep_token == '<eos>':
            added_array = np.concatenate((token_datas, _array.reshape(-1, 1)), axis=1)
        return added_array
    
    def __get_initial_dec_input(self):
        fake_dec_input = []
        for _ in range(len(self.__X)):
            fake_dec_input.append([self.__vocab.wrd2idx('<pad>') for _ in range(self.__max_len)])
        return np.array(fake_dec_input)
    
    def get_vocab(self):
        return self.__vocab


############################
# 레이어 클래스 정의
############################
# 임베딩 레이어
class Embedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.__emb = nn.Embedding(vocab_size, emb_dim)
    
    def forward(self, X):
        return self.__emb(X)

# Attention 레이어
class AttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, bidirectional):
        super().__init__()
        self.__hidden_size = hidden_size
        self.__shape_size = (2 if bidirectional else 1) * n_layers
        
        self.__encoding_layer = nn.LSTM(
            input_size=input_size,      # 입력값의 차원
            hidden_size=hidden_size,    # 히든레이어의 차원
            num_layers=n_layers,        # 각 히든당 레이어의 갯수
            bidirectional=bidirectional,# 양방향 설정
        )
        self.__decoding_layer = nn.LSTM(
            input_size=input_size,      # 입력값의 차원
            hidden_size=hidden_size,    # 히든레이어의 차원
            num_layers=n_layers,        # 각 히든당 레이어의 갯수
            bidirectional=bidirectional,# 양방향 설정
        )
    
    def forward(self, enc_input, dec_input):
        # 초기값 설정
        initial_hidden = torch.zeros(self.__shape_size, enc_input.size(0), self.__hidden_size).to(enc_input.device)
        initial_cell = torch.zeros(self.__shape_size, enc_input.size(0), self.__hidden_size).to(enc_input.device)
        # 인코딩
        enc_output, (enc_hidden, enc_cell) = self.__encoding_layer(enc_input, (initial_hidden, initial_cell))
        n_step = dec_input.size(0)
        hidden_state, cell_state = enc_hidden, enc_cell
        # 스텝의 attention 가중치를 저장할 리스트
        train_attn_weights = []
        # 결과값을 저장할 텐서
        responses = torch.empty(n_step, enc_input.size(0), self.__hidden_size*2).to(enc_input.device)
        # n_step 별 디코딩
        for i in range(n_step):
            # 현재 스텝의 디코딩
            dec_output, (hidden_state, cell_state) = self.__decoding_layer(dec_input[i].unsqueeze(0), (hidden_state, cell_state))
            # 현재 스텝의 가중치 계산
            # weight.shape (batch_size, 1, n_step)
            weight = self.__get_attention_weight(enc_output, dec_output)
            # 현재 스텝의 가중치를 저장
            train_attn_weights.append(weight.squeeze().data.cpu().numpy())
            
            # weight를 이용해 attention value 계산
            # A.shape (a, b, C)
            # B.shape (a, C, d)
            # A.bmm(B).shape -> (a, b, d) (bmm = 배치별 행렬곱)
            
            # weight.shape                     : (batch_size, 1, n_step) 
            # enc_output.transpose(0, 1).shape : (batch_size, n_step, n_hidden)
            # attn_value.shape (batch_size, 1, n_hidden)
            attn_value = weight.bmm(enc_output.transpose(0, 1))
            
            # dec_output과 attn_value의 shape을 맞춰줌
            dec_output = dec_output.squeeze(0) # (, batch_size, n_hidden) -> (batch_size, n_hidden)
            attn_value = attn_value.squeeze(1) # (batch_size, 1, n_hidden) -> (batch_size, n_hidden)
            # dec_output과 attn_value를 연결후 reponses에 저장
            responses[i] = torch.cat((dec_output, attn_value), dim=-1)
            
        # responses.shape (n_step, batch_size, n_hidden*2)
        return responses, train_attn_weights
    
    def __get_attention_weight(self, enc_output, dec_output):
        n_step = enc_output.size(0)
        batch_size = enc_output.size(1)
        # 값을 저장할 텐서 생성
        # attention_score.shape (n_step, batch_size)
        attention_score = torch.zeros((batch_size, n_step)).to(enc_output.device)
        # 각 스텝별 attention score 계산
        for i in range(n_step):
        # 내적을 이용해 attention score 계산
        # enc_output[i].shape (batch_size, n_hidden)
        # dec_output.shape    (batch_size, n_hidden)
            for j in range(batch_size):
                attention_score[j, i] = torch.dot(enc_output[i, j], dec_output[j])
        # attention score를 softmax를 이용해 확률값으로 변환
        # attention_score.shape (batch_size, n_step)
        # attention_score.unsqueeze(1).shape (batch_size, 1, n_step)
        return F.softmax(attention_score, dim=-1).unsqueeze(1)
    
# 아웃풋 레이어
class OutputLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.__linear = nn.Linear(input_size, output_size)
    
    def forward(self, X):
        # X.shape (n_step, batch_size, n_hidden*2)
        n_step  = X.size(0)
        _X = X
        for i in range(n_step):
            # X[i].shape (batch_size, n_hidden*2)
            # _X[i].shape (batch_size, output_size)
            _X[i] = self.__linear(X[i])
        # _X.shape (n_step, batch_size, output_size)
        # _X.transpose(0, 1).shape (batch_size, n_step, output_size)
        return _X.transpose(0, 1)
        
############################
# 모델 클래스 정의
############################
class Ger2EngAttentionModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, bidirectional, emb_type, cell_type='lstm'):
        super().__init__()
        self.__emb_type = emb_type
        self.__cell_type = cell_type
        
        # 임베딩 레이어
        if self.__emb_type == 'emb':
            self.__embedding = nn.Embedding(
                input_size=vocab_size,
                embedding_dim=emb_dim,
            )
        # Attention 레이어
        if self.__cell_type == 'lstm':
            self.__attention = AttentionLSTM(
                input_size=emb_dim,
                hidden_size=hidden_dim,
                n_layers=num_layers,
                bidirectional=bidirectional,
            )
        elif self.__cell_type == 'rnn':
            pass
        # 아웃풋 레이어
        self.__output = OutputLinear(
            input_size=hidden_dim*2,
            output_size=vocab_size,
        )
    
    def forward(self, enc_input, dec_input):
        # 임베딩 레이어
        if self.__emb_type == 'emb':
            enc_input = self.__embedding(enc_input)
            dec_input = self.__embedding(dec_input)
        # Attention 레이어
        dec_output, train_attn_weights = self.__attention(enc_input, dec_input)
        # 아웃풋 레이어
        output = self.__output(dec_output)
        return output, train_attn_weights
    
######################
# 얼리스탑핑 정의
######################
class EarlyStopping():
    def __init__(self, patience=10, save_path=None, target_score=0, model_name='model'):
        # 초기화
        self.best_score = 0
        self.patience_count = 0
        self.target_score = target_score
        self.patience = patience
        self.save_path = save_path
        best_model_name = model_name + '_best.pth'
        self.best_model_path = self.save_path + best_model_name
        last_model_name = model_name + '_last.pth'
        self.last_model_path = self.save_path + last_model_name
    # 얼리 스토핑 여부 확인 함수 정의
    def is_stop(self, model, score):
        # 모델 저장(마지막 모델)
        self.__save_last_model(model)
        # 베스트 스코어가 타겟 스코어보다 낮을 경우
        if self.best_score < self.target_score:
            # 스코어가 이전보다 안좋을 경우
            if self.best_score >= score:
                # patience 초기화
                self.patience_count = 0
                return False
            # 스코어를 업데이트
            self.best_score = score
            # 모델 저장
            self.__save_best_model(model)
            # patience 초기화
            self.patience_count = 0
            return False
            
        
        # 스코어가 이전보다 좋을 경우
        if self.best_score < score:
            # 스코어를 업데이트
            self.best_score = score
            # 모델 저장
            self.__save_best_model(model)
            # patience 초기화
            self.patience_count = 0
            return False
        
        # 스코어가 이전보다 좋지 않을 경우 +
        # 스코어가 타겟 스코어보다 높을 경우
        # patience 증가
        self.patience_count += 1
        # patience가 최대치를 넘을 경우
        if self.patience_count > self.patience:
            return True
        # patience가 최대치를 넘지 않을 경우
        return False
    # 모델 저장 함수 정의
    def __save_best_model(self, model):
        torch.save(model.state_dict(), self.best_model_path)
    # 마지막 모델 저장 함수 정의
    def __save_last_model(self, model):
        torch.save(model.state_dict(), self.last_model_path)
        
        
############################
# 모델 학습 함수 정의
############################
def train(model, loader, optimizer, loss_fn, device):
    model.train()
    train_loss = None
    train_score = None
    for enc_input, dec_input, targets in loader:
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        targets = targets.to(device)
        
        outputs, _ = model(enc_input, dec_input)
        loss = None
        for output, target in zip(outputs, targets):
            loss = loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        
        outputs = F.softmax(outputs, dim=-1)
        targets = targets.detach().cpu().numpy().argmax(axis=-1)
        outputs = outputs.detach().cpu().numpy().argmax(axis=-1)
        scores = []
        for output, target in zip(outputs, targets):
            scores.append(accuracy_score(output, target))
        train_score = np.mean(scores)
    return train_loss, train_score

############################
# 모델 평가 함수 정의
############################
def test(model, loader, device, is_target=False):
    model.eval()
    test_pred = None
    test_score = None
    for enc_input, dec_input, targets in loader:
        enc_input = enc_input.to(device)
        dec_input = dec_input.to(device)
        targets = targets.to(device)
        
        outputs, _ = model(enc_input, dec_input)
        outputs = F.softmax(outputs, dim=-1)
        test_pred = outputs.detach().cpu().numpy()
        
        if is_target:
            targets = targets.detach().cpu().numpy().argmax(axis=-1)
            outputs = outputs.detach().cpu().numpy().argmax(axis=-1)
            scores = []
            for output, target in zip(outputs, targets):
                scores.append(accuracy_score(output, target))
            test_score = np.mean(scores)
    return test_pred, test_score