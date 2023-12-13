import pandas as pd
from sentiment import *
# from extractsum import * 

from kobart import get_kobart_tokenizer
from transformers.models.bart import BartForConditionalGeneration
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
# colab 환경에서
from google.colab import drive


    # predic_sentence = [sentence1, sentence2, sentence3, ...] => [[sentence1, 0], [sentence2, 0], [sentence3, 0], ...]
def predict_emotion(predict_sentence):
    data = []
    for s in predict_sentence:
      data.append([s.replace('\n', ' '), 0])

    #data = [predict_sentence, 0]
    #dataset_another = [data]

    positive_review = []
    negative_review = []

    drive.mount('/content/drive')
    device = torch.device("cuda:0")

    tokenizer_emotion = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
    bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
    vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer_emotion.vocab_file, padding_token='[PAD]')

    tok = tokenizer_emotion.tokenize

    another_test = BERTDataset(data, 0, 1, tok, vocab, 64, True, False)
    test_loader = torch.utils.data.DataLoader(another_test, batch_size=64, num_workers=4)
    
    model_extract_emotion = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model_extract_emotion.load_state_dict(torch.load('/content/drive/MyDrive/AI_project/weight/naver_shopping.pt'))

    model_extract_emotion.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_loader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model_extract_emotion(token_ids, valid_length, segment_ids)
        # test_eval=[]
        for i in range(len(out)):
            logits=out[i]
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                negative_review.append(predict_sentence[64*batch_id + i])

            else:
                positive_review.append(predict_sentence[64*batch_id + i])

    return positive_review, negative_review

def predict_abstract(sentence):
  model_abstract_sum = BartForConditionalGeneration.from_pretrained('./kobart_summary')
  tokenizer_abstract = get_kobart_tokenizer()

  input_ids = tokenizer_abstract.encode(sentence)
  input_ids = torch.tensor(input_ids)
  input_ids = input_ids.unsqueeze(0)
  output = model_abstract_sum.generate(input_ids, eos_token_id=1, max_length=512, num_beams=5)
  output = tokenizer_abstract.decode(output[0], skip_special_tokens=True)

  return output.replace("\n", " ")

def run_baseline(sentence, positive):
  #감정추출 => postive revies, negative review로 분리
  pr, nr = predict_emotion(sentence)

  p_a_sum = ""
  n_a_sum = ""

  if positive == True:

    if len(pr) > 1 :
      p_e_sum = predict_extract(pr)
      p_a_sum = predict_abstract(p_e_sum)

    else :
      p_a_sum = predict_abstract(pr)

    return p_a_sum

  elif positive == False:
    if len(nr) > 1:
      n_e_sum = predict_extract(nr)
      n_a_sum = predict_abstract(n_e_sum)

    else :
      n_a_sum = predict_abstract(nr)

    return n_a_sum


df = pd.read_excel('/content/drive/MyDrive/AI_project/data/oliveyoung_1.xlsx' )
sentence = df['리뷰 본문'].tolist()

n_sum= run_baseline(sentence, False)
print(n_sum)

p_sum= run_baseline(sentence, True)
print(p_sum)