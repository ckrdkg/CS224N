# Lecture 20. Future of NLP + Deep Learning

* Chris is still traveling
* Kevin Clark who's one of the PhD students in the NLP lab.

![image](https://user-images.githubusercontent.com/34912004/102221471-8d2fec00-3f25-11eb-9a69-76d81eda4f66.png)

![image](https://user-images.githubusercontent.com/34912004/102223061-90c47280-3f27-11eb-9ac6-26113b85060f.png)

##### Why has deep learning been so successful recently?
![image](https://user-images.githubusercontent.com/34912004/102226521-d2efb300-3f2b-11eb-899f-fbe7a4cffd55.png)

1980~1990년대에 neural network에 관한 연구가 활발하게 진행되었음
increse the size of data, the size of the models -> accuracy 향상

##### Big deep learning successes
![image](https://user-images.githubusercontent.com/34912004/102330964-8e1c5880-3fcd-11eb-9e7a-416638b1af8b.png)

Image Recognition:
ImageNet: 14 million images

Machine Translation:
WMT: Millions of sentence pairs

Game Playing:
10s of millions of frames for Atari AI
10s of millions of self-play games for AlphaZero

![image](https://user-images.githubusercontent.com/34912004/102333390-b35e9600-3fd0-11eb-85ea-a0e18ccddac9.png)

problem: 영어가 모국어인 인구는 세계 인구의 10% 미만임에도 불구하고 대부분의 datase은 영어로 이루어짐
(하지만 우리는 딥러닝 스케일과 더욱 큰 모델을 만들고 싶다.)
solution: Using unlabeled data

# 1. Using Unlabeled Data for Translation

Using unlabeled data to improve machine translation models.

##### 가. Pre-training (it's really reminiscent of ELMO)
![image](https://user-images.githubusercontent.com/34912004/102334663-49df8700-3fd2-11eb-9d37-7bdd5cb1948f.png)

- encoder, decoder를 먼저 따로 train
- 2가지를 같이 train
- encoder를 source side language에 대해 train된 언어 모델의 가중치로 초기화
- decoder를 target size language에 대해 train된 가중치로 초기화
  
이 모델은 모델 performance를 향상시킨다

![image](https://user-images.githubusercontent.com/34912004/102335628-819afe80-3fd3-11eb-905a-3db0f491f7e9.png)

But,
problem: 2가지 언어 사이에는 어떤 interaction도 없다.
다시 말해, I traveled to Belgium 이라는 문장을 translation하면 target값이 나오지만 정확하지 않을 수가 있다.
solution: Self-Training, Back-Translation

###### 1) Self-Traning
![image](https://user-images.githubusercontent.com/34912004/102345537-e4df5d80-3fe0-11eb-8960-d52decbacfce.png)

- labeled data로 train해서 unlabeled data의 label값 예측
- 그 중 가장 확률값이 높은 data들만 labeled data로 가져가서 retrain
  
###### 2) Back-Translation
This technique is really a very popular.
![image](https://user-images.githubusercontent.com/34912004/102346014-967e8e80-3fe1-11eb-88ab-fdc3c1bf018d.png)

switch the source side and target side.
즉 두번째 모델 돌릴땐 French sentence가 source

왜 이 방법이 더 나을까?
- There's no longer circularity
- Models never see “bad” translations, only bad inputs

![image](https://user-images.githubusercontent.com/34912004/102362292-8ec9e480-3ff7-11eb-85a8-bffe08282204.png)

###### 3) what if there is no Bilingual Data?
Assumtion: 
1. monolingual corpora만 가지고 있고, human translation한 문장은 없음.
2. 두 가지 언어로 된 문장만 가지고 있음.
3. 외계인이 와서 말을 걸기 시작한다.
"외계인이 말하는 것을 영어로 번역할 수 있을까?

###### 4) Unsupervised Word Translation
가) word to word translation

목표: 한가지 언어의 word가 주어졌을 때, labeled data를 사용하지 않고 translation을 하는 것

![image](https://user-images.githubusercontent.com/34912004/102367175-cb4c0f00-3ffc-11eb-85dc-420161f58999.png)

(Goal: Learn word vectors for word in both languages)
English -> German
embedding space 공간에서 영어 단어를 하나 고르면 그 단어와 가장 가까운 위치의 단어를 찾으면 됨. 

(근데 단어들이 워낙 많으니까 가깝다고 해서 그게 꼭 같은 의미일까? 라는 의문이 생김 그래서 가정을 함)

• Assumtion
- 구조는 언어마다 비슷해야 합니다.

이 embedding space의 구조는 regularity를 가지고 있고 우리는 그 regularity을 이용하여 이 사이의 정렬을 찾을 수 있습니다.

![image](https://user-images.githubusercontent.com/34912004/102368209-f3883d80-3ffd-11eb-86c6-ba4056a91803.png)

(왼쪽 사진을 보면 두 구조는 다르지만 비슷함을 확인할 수 있음. cat & feline, gatto & felino)

orthogonal matrix W는 X를 회전하는 용도로 사용

Question?
W matrix를 직교로 하는 이유?
성능이 저하되고 overfitting을 피하기 위함.

![image](https://user-images.githubusercontent.com/34912004/102368947-c5efc400-3ffe-11eb-9c04-02f6c7c683ad.png)

adversarial training: regularization의 한 방법. train된 모델에 대하여 input data를 잘못 예측하도록 input을 조작하는 것

Discriminator: embedding이 Y에서 온건지 X에서 변환된 Wx인지 예측(즉, 빨간점인지 파란점인지)

2가지 언어가 완전히 분리되어 있기 때문에 W matrix가 없다면 discriminator에겐 쉬운 일이지만 W matrix가 있다면 discriminator는 좋은 결과를 낼 수 없다.
W matrix를 훈련시키는 이유는 discriminator를 최대한 confuse하게 만들기 위함.

##### 나) full sentence to sentence translation

###### 1) de-noising autoencoder
![image](https://user-images.githubusercontent.com/34912004/102471849-0734b200-4099-11eb-9092-50c8310e5b0d.png)

- Input Data에 random하게 noise를 추가하고 noise가 없는 원본 input을 재구성하도록 학습시키는 네트워크
- noise를 추가하더라도 manifold상에서는 똑같은 곳에 분포된다는 가정이 존재

##### 다) Performance
![image](https://user-images.githubusercontent.com/34912004/102474275-ca1def00-409b-11eb-8ea5-9675976df750.png)

* sentences가 증가할수록 Supervised translation model이 더 좋음
* 하지만 sentece가 10000개에서 100000개 사이라면 Unsupervised translation model이 더 좋음

예시) Attribute Transfer
hashtag를 사용하며 "relaxed", "annoyed"를 포함하는 tweets을 수집
Learn unsupervised model해서 문장의 다른 단어들을 보존하면서 relaxed와 annoyed를 서로 변환

![image](https://user-images.githubusercontent.com/34912004/102475879-c4c1a400-409d-11eb-9b28-5e2e5024d291.png)

# 2. Cross-Lingual BERT
![image](https://user-images.githubusercontent.com/34912004/102706565-574c8800-42d6-11eb-9104-690289eb6837.png)

가. 기존의 BERT 모델
- 일부 단어를 masking하고 예측

나. Facebook이 propose한 새로운 종류
- input에 english와 french 모두 주고 masking 후 예측
- 동기: 두 언어 사이의 relation을 더 잘 이해하게 하기 위함


# 3. Huge Models and GPT-2
##### 가. General Trend in ML
![image](https://user-images.githubusercontent.com/34912004/102482617-d491b600-40a6-11eb-8fcf-7b93e74cd81c.png)

Huge Model in Computer Vision
![image](https://user-images.githubusercontent.com/34912004/102483092-9779f380-40a7-11eb-9b61-51e843d4670c.png)
- GAN으로 생성한 hallucinated image
- huge model들이 image recognition에 사용되고 있음

Recent work by Google
![image](https://user-images.githubusercontent.com/34912004/102483632-5d5d2180-40a8-11eb-84c9-1ac26805380c.png)
- dir 5억 개의 매개 변수가 있는 이미지 네트 모델
- 파라미터의 수가 증가함에 따라서 accuracy가 증가하고 있음

Training Huge Model
- 당연하게도 더 좋은 hardware가 필요
- Data and Model parallelism
  Data parallelism: 16개의 GPU와 각각의 GPU 당 batch size가 32라고 가정한다면 더 빨리 훈련시킬 수 있다.
  Model parallelism: model이 너무 컸을 때, model들을 분할하여 여러 대의 컴퓨터가 수행하도록 병렬화
  ![image](https://user-images.githubusercontent.com/34912004/102709144-e4014100-42ea-11eb-8585-3a38167fecc5.png)


##### 나. GPT-2
- 이전의 알고리즘들과의 다른 점은 매우 크다는 것
- 상당히 많은 양의 텍스트로 훈련되어 있다.(약 40기가)
- 기존의 벤치마크에서 실행 가능(벤치마크에 대한 train data를 보지 못하더라도 성능이 좋은듯)
- Penn Treebank에서 train했을 때의 벤치마크
 ![image](https://user-images.githubusercontent.com/34912004/102495664-b71a1780-40b9-11eb-9993-4bd0c5be180a.png)

How can GPT-2 be doing translation?
![image](https://user-images.githubusercontent.com/34912004/102503057-ad48e200-40c2-11eb-9b1d-64deb8e074c6.png)
여러 페이지로 나누어 주고 영어로 이루어져 있는데도 translation을 굉장히 잘한 상태

GPT-2 Question Answering
![image](https://user-images.githubusercontent.com/34912004/102503767-66a7b780-40c3-11eb-8952-eafbf9d10257.png)

What happens as models get even bigger?
![image](https://user-images.githubusercontent.com/34912004/102515537-3b2bc980-40d1-11eb-85cb-9e641333636d.png)
강사는 parameter가 1조개가 된다면 인간 수준에 도달할 것이라고 예상

But, trend isn't clear.
![image](https://user-images.githubusercontent.com/34912004/102515849-9362cb80-40d1-11eb-817c-965e955e1133.png)
성능이 이미 최고 수준에 이른 것 같아서 앞으로 흥미로운 일이 될 것 같다.

GPT-2 Reaction
![image](https://user-images.githubusercontent.com/34912004/102578910-a65fb500-413e-11eb-9dc6-a284fbb60aaf.png)
- release 반대 의견쪽에서 이게 그렇게 특별한건가 이 모델이 release되지 않아도 5년 안에 사람들이 더 좋게 할 수 있을 것이고 포토샵으로도 fake image를 만들어 낼 수 있다 등의 의견이 나왔고

- 그런 반면에 포토샵으로 그런 작업들을 할 순 있지만 fake reviews, news, comment들의 위험성이 있고, 이미 광범위하게 존재한다.

![image](https://user-images.githubusercontent.com/34912004/102581298-9bf3ea00-4143-11eb-80f6-401d48072a17.png)
어찌됐든 NLP는 점점 High-Impact Decision을 하고 있다(재판, 고용, 채점 등)
대표적인 예시로,

![image](https://user-images.githubusercontent.com/34912004/102581375-c645a780-4143-11eb-8b30-330b27ae23d2.png)
아마존의 AI가 성차별적인 것으로 드러났다던지, AI가 점점 사람들을 감옥으로 보내고 있다든지의 예시가 있다. (bad results)

또 다른 예시로 chatbot이 있다.
![image](https://user-images.githubusercontent.com/34912004/102581868-c2feeb80-4144-11eb-95a1-74fc9d148a70.png)

Woebot은 사람의 기분이 좋지 않을 때 대화할 수 있는 기능이 있다. 어쩌면 사람들을 도울 수 있는 좋은 기술이 될 수 있다.

다른 한편으로는 리스크가 존재한다. 마이크로소프트에서 트위터에 대한 챗봇을 train시켰는데 인종차별주의적, 성차별적 등의 편견들을 배웠다.

NLP가 점점 효과적으로 변해가면서 좋은 점도 있지만 그에 따른 리스크도 존재한다.

# 4. What did BERT "solve" and what do we work on next?

##### 가. The Death of Architecture Engineering?

![image](https://user-images.githubusercontent.com/34912004/102582941-035f6900-4147-11eb-998e-ba1401d0f02c.png)

reseracher의 관점에서 6개월 동안 연구해서 1 F1 point를 얻을 수 있고 아니면 BERT를 3배 키워서 5 F1 point를 얻을 수도 있다.

실제로 SQuAD 리더보드의 Top 20은 모두 BERT를 사용했다.

##### 나. Harder Natural Language Understanding
그래서 BERT에게는 다음과 같은 더 어려운 작업이 필요하다.
![image](https://user-images.githubusercontent.com/34912004/102584906-b1b8dd80-414a-11eb-8362-585d32f84cc5.png)

###### 1) QuAC(Question Answering in Context)
![image](https://user-images.githubusercontent.com/34912004/102585025-f17fc500-414a-11eb-913a-b783eafea793.png)

앞의 대화 내용을 읽어보거나 주제가 Daffy Duck인 것을 모르는 한 3번째 질문에 대답할 수 없다.

![image](https://user-images.githubusercontent.com/34912004/102585055-ffcde100-414a-11eb-9d2d-94ccb01aa537.png)

QuAC에서의 BERT와 human performance는 큰 차이가 난다.

###### (2) HotpotQA
![image](https://user-images.githubusercontent.com/34912004/102585583-fb55f800-414b-11eb-8a80-3fc3dd616dad.png)

- multiple documents들을 보고 추론해야 한다.

![image](https://user-images.githubusercontent.com/34912004/102585691-2f311d80-414c-11eb-9908-dc3713a5ee34.png)

- QuAC에서의 human performance보다 더 큰 차이가 난다.

###### 3) Multi-Task Learning
(NLP의 다른 영역은 한 모델이 많은 task들을 수행하도록 하는 것)

![image](https://user-images.githubusercontent.com/34912004/102586925-5557bd00-414e-11eb-90fd-271d63b24114.png)

성능 측면에서 BERT로 Multi-task한 방식이 BERT보다 뛰어나다.

###### 4) Another Area
- low-resource 환경에 대처하는 것
  : 컴퓨터 성능 등의 자원(BERT를 실행하는 데 필요한 컴퓨팅)
- Interpreting/Understanding Models
  : 일부 APP(의료 등)에 중요
  ex) 단순히 환자의 병을 진단해주는게 아니라 어떤 이유로 병을 앓고 있는지 검증하는 것

# 5. Conclusion
![image](https://user-images.githubusercontent.com/34912004/102589919-15df9f80-4153-11eb-8abe-0ba33df1a5b5.png)
