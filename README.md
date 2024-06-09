
# Fake News Detection Project
___

<div align="center">
</div>


## Fake News Detection with Language Models(KoBERT).
> **Personal Project** , **Jan. 2024 ~ Feb. 2024**

---



## Software Stacks
![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)




---

## Project Motivation

- The amount of information has increased due to the development of information production technology.
- This has led to an increase in the production of **fake information**.
- Fake information can confuse readers and spread misinformation, which can lead to a variety of problems.
- we study the **detection of “click-bait”**, one of the types of fake news.
---

## Implementation

### 1. Preprocessing Data
- Before tokenize, add "Special Token" between title and context
<img width="450" alt="image1" src = "https://github.com/user00144/fake_news_detection/assets/120700820/400d49a7-2bc2-443e-8304-a8a27bab1192">

### 2. DL Model Design
- Using "Pre-trained Ko-BERT Model"
- Ko-BERT Model([link](https://github.com/SKTBrain/KoBERT))
<img width="450" alt="image1" src = "https://github.com/user00144/fake_news_detection/assets/120700820/b25f6c2e-99ea-4407-be44-be15fa49c48f">


### 3. Training Result

| loss | accuracy |
|---|---|
|0.3304|98.05 %|

### 4. Crawling and inferring real-world news articles
- We crawled 355 "Naver News Articles", Inference with our model.
- The result is shown below
<img width="346" alt="image" src="https://github.com/user00144/fake_news_detection/assets/120700820/28f5f915-4e46-4b18-bbc2-55e0bb7053fd">


---

## Outputs

- **Publication Conference Paper** in Korean Society For Internet Information Spring Conference (Apr. 2024)
<img width="820" alt="스크린샷 2024-06-09 오후 4 16 38" src="https://github.com/user00144/fake_news_detection/assets/120700820/5df91b30-befa-4063-a2b9-57ec09a19819">
