#####################
# 모듈 임포트
#####################
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

#####################
# vocab 클래스
#####################
class AlphabetVocab():
    def __init__(self):
        self.__idx2char = 'SEPabcdefghijklmnopqrstuvwxyz'
        self.__char2idx = {c: i for i, c in enumerate(self.__idx2char)}

    def encode(self, word):
        return [self.__char2idx[c] for c in word]

    def decode(self, idx):
        return ''.join([self.__idx2char[i] for i in idx])
    
    def __len__(self):
        return len(self.__idx2char)
    
    def idx2char(self, idx):
        return self.__idx2char[idx]
    
    def char2idx(self, char):
        return self.__char2idx[char]
    
#####################
# 데이터셋 클래스
#####################
class ConverseWordDataset(Dataset):
    ######################
    # 데이터셋 필수 함수 정의
    ######################
    def __init__(self, X, y=None, max_len=10, vocab=None, embedding_type='emb'):
        super().__init__()
        # 모드 저장
        self.is_train_mode = True if y is not None else False
        self.embedding_type = embedding_type
        # 최대 길이, vocab 저장
        self.max_len, self.vocab = max_len, vocab
        # one hot encoding vector
        self.oh_vector = np.eye(len(self.vocab))
        # 데이터를 토큰화
        self.X = self.__tokenizer(X)
        # y가 있으면 토큰화
        if y is not None:
            _y = self.__tokenizer(y)
            # 디코더의 입력값과 출력값을 구분하기 위해 시작 토큰과 끝 토큰을 추가
            self.y_dec = self.__add_se_token(_y)
            self.y_target = self.__add_se_token(_y, sep_tok='E')
            # 평가 모드일 경우 사용할 가짜 입력값 생성
            self.y_fake = self.__get_eval_dec_input()
        else:
            # y가 없으면 디코더의 가짜 입력값만 생성
            # y_target을 None으로 반환하면 오류 -> 그냥 y_fake동일하게 생성
            self.y_dec = None
            self.y_target = self.__get_eval_dec_input()
            self.y_fake = self.__get_eval_dec_input()
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 임베딩 타입이 emb일 경우
        if self.embedding_type == 'emb':
            return self.__getitem_emb(idx)
        # 임베딩 타입이 ohe일 경우
        elif self.embedding_type == 'ohe':
            return self.__getitem_ohe(idx)
    
    def __getitem_emb(self, idx):
        # 인코더의 입력값을 tensor로 변환
        feature = torch.tensor(self.X[idx], dtype=torch.int32)
        # y값이 있고 학습 모드일 때
        if self.y_target is not None and self.is_train_mode:
            # 디코더의 입력값(SE토근이 추가된 y)을 tensor로 변환
            dec_input = torch.tensor(self.y_dec[idx], dtype=torch.int32)
            # 타겟값을 ohe 후 tensor로 변환 -> pred값과 비교하기 위해 ohe
            # loss 계산을 위해 tensor.float으로 변환
            target = self.oh_vector[self.y_target[idx]]
            target = torch.tensor(target, dtype=torch.float)
            # feature, dec_input, target을 반환
            return feature, dec_input, target
        # y값이 없거나 평가 모드일 때
        else:
            # 디코더의 입력값(SPPPP...)을 tensor로 변환
            dec_input = torch.tensor(self.y_fake[idx], dtype=torch.int32)
            # 타겟값을 ohe 후 tensor로 변환 -> pred값과 비교하기 위해 ohe
            # loss 계산을 위해 tensor.float으로 변환
            target = self.oh_vector[self.y_target[idx]]
            target = torch.tensor(target, dtype=torch.float)
            # feature, dec_input을 반환, target은 있을경우 반환(없으면 dec_input과 동일)
            return feature, dec_input, target
        
    def __getitem_ohe(self, idx):
        # 인코더의 입력값을 ohe
        feature = self.oh_vector[self.X[idx]]
        # tensor로 변환
        feature = torch.tensor(feature, dtype=torch.float)
        if self.y_target is not None and self.is_train_mode:
            # 디코더의 입력값과 출력값(SE토근이 추가된 y)을 ohe
            dec_input = self.oh_vector[self.y_dec[idx]]
            target = self.oh_vector[self.y_target[idx]]
            # tensor로 변환
            dec_input = torch.tensor(dec_input, dtype=torch.float)
            target = torch.tensor(target, dtype=torch.float)
            return feature, dec_input, target
        else:
            # 디코더의 입력값만 ohe (SPPPP...)
            dec_input = self.oh_vector[self.y_fake[idx]]
            target = self.oh_vector[self.y_target[idx]]
            # tensor로 변환
            dec_input = torch.tensor(dec_input, dtype=torch.float)
            target = torch.tensor(target, dtype=torch.float)
            return feature, dec_input, target
    
    ######################
    # 데이터셋 내부 함수 정의
    ######################
    # 모드 변경 함수
    def train(self):
        if self.y_target is None:
            raise Exception('y값이 없습니다.')
        self.is_train_mode = True
    def eval(self):
        self.is_train_mode = False
    
    # 토큰화
    def __tokenizer(self, datas):
        token_datas = []
        for data in datas:
            # 패딩
            padding_data = data + 'P' * (self.max_len - len(data))
            # 토큰화
            token_data = self.vocab.encode(padding_data)
            token_datas.append(token_data)
        # 토큰화된 데이터를 numpy 배열로 변환
        return np.array(token_datas)
    
    # 시작 토큰과 끝 토큰을 추가
    def __add_se_token(self, token_datas, sep_tok='S'):
        add_sep_datas = []
        for token_data in token_datas:
            if sep_tok == 'S':
                add_sep_data = [self.vocab.char2idx(sep_tok)] + list(token_data)
            elif sep_tok == 'E':
                add_sep_data = list(token_data) + [self.vocab.char2idx(sep_tok)]
            add_sep_datas.append(add_sep_data)
        return np.array(add_sep_datas)
    
    # 디코더의 기본 입력값만 생성
    def __get_eval_dec_input(self):
        # 데이터의 총 갯수
        data_len = len(self.X)
        # max길이의 S로 시작해 P가 이어지는 데이터 생성 * 데이터의 총 갯수
        eval_dec_input = [[self.vocab.char2idx('S')] + [self.vocab.char2idx('P')] * (self.max_len)\
                                                                                for _ in range(data_len)]
        return np.array(eval_dec_input)

######################
# 레이어 정의
######################
# 임베딩 레이어
class ConvWordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,     # 임베딩할 단어의 갯수
            embedding_size, # 임베딩 결과의 차원
        )
    
    def forward(self, X):
        return self.embedding(X)

# Seq2Seq 레이어 (RNN)
class ConvWordSeq2SeqRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bidirectional=False):
        super().__init__()
        self.enc_input_size = input_size
        self.hidden_size = hidden_size
        self.shape_size = (2 if bidirectional else 1) * n_layers

        self.encoding_layer = nn.RNN(
            input_size=input_size,      # 입력값의 차원
            hidden_size=hidden_size,    # 히든레이어의 차원
            num_layers=n_layers,        # 각 히든당 레이어의 갯수
            batch_first=True,           # 배치 사이즈를 맨 앞으로 설정
            bidirectional=bidirectional,# 양방향 설정
        )
        self.decoding_layer = nn.RNN(
            input_size=input_size,      # 위와 동일
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, enc_X, dec_X):
        # 기본 히든 벡터의 모양을 정의
        shape = (self.shape_size, enc_X.size(0), self.hidden_size)
        # 기본 히든 벡터를 생성
        hidden_state = torch.zeros(shape).to(enc_X.device)
        # 인코더의 입력값을 넣어 인코딩
        _, enc_hidden = self.encoding_layer(enc_X, hidden_state)
        # 인코더의 히든 벡터를 디코더의 히든 벡터로 사용
        dec_output, _ = self.decoding_layer(dec_X, enc_hidden)
        return dec_output

# Seq2Seq 레이어 (LSTM)
class ConvWordSeq2SeqLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, bidirectional=False):
        super().__init__()
        self.enc_input_size = input_size
        self.hidden_size = hidden_size
        self.shape_size = (2 if bidirectional else 1) * n_layers

        self.encoding_layer = nn.LSTM(
            input_size=input_size,      # 입력값의 차원
            hidden_size=hidden_size,    # 히든레이어의 차원
            num_layers=n_layers,        # 각 히든당 레이어의 갯수
            batch_first=True,           # 배치 사이즈를 맨 앞으로 설정
            bidirectional=bidirectional,# 양방향 설정
        )
        self.decoding_layer = nn.LSTM(
            input_size=input_size,      # 위와 동일
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, enc_X, dec_X):
        # 기본 히든 벡터의 모양을 정의
        shape = (self.shape_size, enc_X.size(0), self.hidden_size)
        # 기본 히든 벡터, 셀 벡터를 생성
        hidden_state = torch.zeros(shape).to(enc_X.device)
        cell_state = torch.zeros(shape).to(enc_X.device)
        # 인코더의 입력값을 넣어 인코딩
        _, (enc_hidden, enc_cell) = self.encoding_layer(enc_X, (hidden_state, cell_state))
        # 인코더의 히든 벡터를 디코더의 히든 벡터로 사용
        dec_output, (_, __) = self.decoding_layer(dec_X, (enc_hidden, enc_cell))
        return dec_output

# 출력 레이어
class ConvWordOutput(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.output_layer = nn.Sequential(
            nn.Linear(input_size, output_size), # 입력값의 차원, 출력값의 차원
            # nn.Softmax(dim=-1),                 # 출력값을 확률로 변환
        )
        
    def forward(self, X):
        return self.output_layer(X)

######################
# 모델 정의
######################
class ConvWordModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, n_layers=1, bidirectional=False, embedding_type='emb', s2s_cell='rnn'):
        super().__init__()
        num_bi = 2 if bidirectional else 1
        self.embedding_type = embedding_type
        # 임베딩 레이어
        if embedding_type == 'emb':
            self.embedding_layer = ConvWordEmbedding(
                vocab_size=vocab_size,         # 임베딩할 단어의 갯수
                embedding_size=embedding_size,  # 임베딩 결과의 차원
            )
        # Seq2Seq 레이어
        if s2s_cell == 'rnn':
            self.seq2seq_layer = ConvWordSeq2SeqRNN(
                input_size=embedding_size,     # 인코더, 디코더의 입력값의 차원
                hidden_size=hidden_size,        # 히든 레이어의 차원
                n_layers=n_layers,              # 각 히든당 레이어의 갯수
                bidirectional=bidirectional,    # 양방향 설정
            )
        elif s2s_cell == 'lstm':
            self.seq2seq_layer = ConvWordSeq2SeqLSTM(
                input_size=embedding_size,     # 인코더, 디코더의 입력값의 차원
                hidden_size=hidden_size,        # 히든 레이어의 차원
                n_layers=n_layers,              # 각 히든당 레이어의 갯수
                bidirectional=bidirectional,    # 양방향 설정
            )
        # 출력 레이어
        self.output_layer = ConvWordOutput(
            input_size=num_bi * hidden_size, # 입력값의 차원(아웃풋 벡터의 차원)
            output_size=vocab_size,         # 출력값의 차원(num_classes)
        )
    
    def forward(self, X, y):
        if self.embedding_type == 'emb':
            # 인코더의 입력값을 임베딩
            enc_input = self.embedding_layer(X)
            # 디코더의 입력값을 임베딩
            dec_input = self.embedding_layer(y)
        elif self.embedding_type == 'ohe':
            enc_input = X
            dec_input = y
        # print(enc_input.shape, dec_input.shape)
        # 인코더와 디코더의 입력값을 넣어 dec_output 생성
        dec_output = self.seq2seq_layer(enc_input, dec_input)
        # dec_output을 출력값으로 변환
        output = self.output_layer(dec_output)
        return output
    
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

######################
# 트레인 루프 정의
######################
def train(model, loader, optimizer, device, loss_fn):
    # 모델을 학습 모드로 설정
    model.train()
    # loss를 저장할 변수
    loss_score = None
    # score를 저장할 변수
    acc_score = None
    # Softmax를 적용하기 위해 모델 생성
    softmax = nn.Softmax(dim=-1)
    # 학습 루프
    for feature, dec_input, target in loader:
        # device에 데이터 이동
        feature = feature.to(device)
        dec_input = dec_input.to(device)
        target = target.to(device)
        # 모델에 feature와 dec_input을 넣어 예측값 생성
        output = model(feature, dec_input)
        # print(output.shape, target.shape)
        # (batch_size, seq_len, num_classes) 끼리 비교하면 잘됨
        # loss 계산
        for o, t in zip(output, target):
            loss = loss_fn(o, t)
        # loss 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # softmax 적용
        output = softmax(output)
        # loss 기록
        loss_score = loss.item()
        target = target.detach().cpu().numpy().argmax(axis=-1)
        output = output.detach().cpu().numpy().argmax(axis=-1)
        # score 계산
        scores = []
        for t, o in zip(target, output):
            scores.append(accuracy_score(t, o))
        # score 기록
        acc_score = np.mean(scores)
    # loss와 score를 반환
    return loss_score, acc_score

######################
# 테스트 루프 정의
######################
@torch.inference_mode()
def test(model, loader, device, is_target=False):
    # 모델을 평가 모드로 설정
    model.eval()
    # 예측값을 저장할 변수
    pred = None
    # acc를 저장할 변수
    acc_score = None
    # Softmax를 적용하기 위해 모델 저장
    softmax = nn.Softmax(dim=-1)
    # 평가 루프
    for feature, dec_input, target in loader:
        # device에 데이터 이동
        feature = feature.to(device)
        dec_input = dec_input.to(device)
        # 모델에 feature와 dec_input을 넣어 예측값 생성
        output = model(feature, dec_input)
        # softmax 적용
        output = softmax(output)
        # 예측값을 pred에 저장
        pred = output.detach().cpu().numpy()
        # target이 있으면 acc 계산
        if is_target:
            # target을 cpu로 이동 후 numpy로 변환
            target = target.detach().cpu().numpy().argmax(axis=-1)
            # output을 cpu로 이동 후 numpy로 변환
            output = output.detach().cpu().numpy().argmax(axis=-1)
            # acc 계산
            acc_scores = []
            for t, o in zip(target, output):
                acc_scores.append(accuracy_score(t, o))
            # acc 기록
            acc_score = np.mean(acc_scores)
    # 예측값과 acc를 반환
    return pred, acc_score