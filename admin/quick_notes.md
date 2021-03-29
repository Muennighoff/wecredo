25/01 - Tencent Meeting

Two Things working on:
- Multimodal model
- Build an AutoML extractor

Working: 
- Usually discuss possible progress of the week
- Offers his assistance 

3 things:
- Otolion & Sugon: Working on DL for tabular data
    - Lots of problems with TabNet - torch & tf implementations are wrong
    - Get preliminary results for new proposed structures before CNY
- Dr. Luo: Autodriving pipeline; Use RL to increase sample efficiency
    - In easier tasks like OpenAI Gym, you don't have a lot of challenge
- Me: NLP side

By the end of March: NLP Lab with Xjinhua set up

End goal: Research-oriented

Wang Ran: Mathematical background


NLP:
- Switch Transformers
- Whole Word Masking - Need a good tokenizer
    - Free tokenizers are bad
    



- 2 GCloud Accounts
    - 1 from DeepMind with cheaper costs for big trainings - will not give me access - have to ask; for research only
    - 1 from the company; access
    - Use Colab as much as poss (> Kaggle)
    - Tell him if use GCloud; Normally 1 GPU on GCloud should be enough; V2 (?) GPU

- Make decisions & ideas on my own

- Don't work 9-9; No need to start at specific time, just send message
- Send plan & time limit / deadlines of my work

- Prefer PyTorch, but TFlow 2.3 for Project to use TPUs (2.4 strange error)

- Not let too many people know

- Support competitions, but not paid during time working on competition; Prize money split among participants; Company will use it for publicity
    - Can apply for cloud budget

- Read papers on LayerNorm etc; Plan to use shared weights like in Albert, but different weights for en & decoder; Seq-to-Seq Model; T5-like Model as best on GLUE

- Uploads data to GBucket

- Two current aproaches
    - Use the best CN Model from Google, which is trained on CN Wikipedia (CN Wikipedia sucks)
    - Train own one like the ENG ones


- Group Meetings every Monday - No need to prepare any slides, just share what you've been working on



Questions:

- So there are two approaches we consider right now:
    - a) We use the CN model from Google which is trained on Wikipedia
    - b) We train our own large-scale CN model on CommonCrawl Data
        - In this case we would pre-train it  

    -> We will train a new model using Performer Structure & T5 Training Strategy; Existing CN models are bad


- You mentioned about seq-to-seq tasks, such as translation. But the final task of predicting whether a company will default will be a goal of binary prediction, no? Where do we need seq-to-seq?
    -> To check whether pre-training was successful

