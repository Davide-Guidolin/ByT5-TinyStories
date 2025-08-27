# ByT5-TinyStories

This project is an exploration into building, training, and fine-tuning a byte-level language model from scratch following the [ByT5](https://arxiv.org/abs/2105.13626) architecture.

## Project Goals

The primary goal is to learn and implement key concepts in modern NLP by going through a complete project lifecycle:

1.  **Implement a Transformer:** Build a encoder-decoder, byte-level transformer language model, based on the principles of the [ByT5 paper](https://arxiv.org/abs/2105.13626).
2.  **Train on TinyStories:** Train the model on the `roneneldan/TinyStories` dataset to generate coherent short stories.
3.  **Fine-Tune with LoRA:** Implement and apply Low-Rank Adaptation (LoRA) to efficiently fine-tune the base model on a specific theme or style of story.

## Why Byte-Level?

While most state-of-the-art models use tokenizers to operate on word or sub-word pieces, this project intentionally uses a byte-level approach for several key reasons:

*   **Simplicity:** It is the most fundamental representation of text. There is no need to train or manage a separate tokenizer, which simplifies the data pipeline.
*   **No Out-of-Vocabulary (OOV) Tokens:** The model can, in theory, represent any text, including typos, rare words, or even multiple languages, without resorting to unknown (`<UNK>`) tokens.
*   **Learning Word Structure:** By operating on bytes, the model is forced to learn the structure of words, prefixes, and suffixes from the ground up, which is a valuable learning exercise.

## Notes

The whole project has been built with [Gemini CLI](https://github.com/google-gemini/gemini-cli) on my side, as an assistant. Gemini did not modify my files, it only acted as a teacher.

The flow during the project was: 
1. implement
2. ask Gemini if the code was ok
3. ask for doubts / clarifications
4. repeat

# Model specs
TODO

# Training

Training has been performed on an L40S GPU on [Runpod](https://www.runpod.io/).

## First Run
Wandb Logs: [Run 1](https://wandb.ai/dg11/byt5-tinyStories/runs/1grebfvl?nw=nwuserdavideguidolin11)

* Model Size: `small`
* Effective batch size (batch_size * accumulation_steps * block_size): `2^18`
* Max learning rate: `6e-4`
* Warmup steps: `1000`
* Seed: `42`
* Total cost: `7.34 $`

I'm overall happy with the first run, loss went down faster than expected and it was satisfying to see the model starting to create meaningful words.
I stopped the run at ~4500 optimization steps, after the gradient norm started to have some unusual spikes. Over the entire run the output contained non-printable bytes.
This is text generated at ~3900 optimization steps, before the gradient spikes:
```
� to the tree and saw a lot of fun�t it was very fast. He had fun and � made the pigeon and looked a�d to play with his toys and b� felt very happy. She said they co�d and asked, "What is that? I�and said, "I want to [EOS]
```

I decided to stop the run due to the gradient norm spikes (probaby learning rate too high?) and to investigate the non-printable bytes. Turned out these are sentinel tokens (used for the loss) whose probability should be set to zero during generation.

## TODO
* Read about attention-dropout
* Implement KV Caching for efficient generation