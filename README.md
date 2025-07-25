# TinyStories-LoRA: A Character-Level Language Model

This project is an exploration into building, training, and fine-tuning a character-level language model from scratch using a transformer architecture.

## Project Goals

The primary goal is to learn and implement key concepts in modern NLP by going through a complete project lifecycle:

1.  **Implement a Transformer:** Build a encoder-decoder, character-level transformer language model, based on the principles of the [ByT5 paper](https://arxiv.org/abs/2105.13626).
2.  **Train on TinyStories:** Train the model on the `roneneldan/TinyStories` dataset to generate coherent short stories.
3.  **Fine-Tune with LoRA:** Implement and apply Low-Rank Adaptation (LoRA) to efficiently fine-tune the base model on a specific theme or style of story.

## Why Character-Level?

While most state-of-the-art models use tokenizers to operate on word or sub-word pieces, this project intentionally uses a character-level approach for several key reasons:

*   **Simplicity:** It is the most fundamental representation of text. There is no need to train or manage a separate tokenizer, which simplifies the data pipeline.
*   **No Out-of-Vocabulary (OOV) Tokens:** The model can, in theory, represent any text, including typos, rare words, or even multiple languages, without resorting to unknown (`<UNK>`) tokens.
*   **Learning Word Structure:** By operating on characters, the model is forced to learn the structure of words, prefixes, and suffixes from the ground up, which is a valuable learning exercise.
*   **Robustness:** Character-level models can be more robust to noise and variations in the input text.


## TODO
* Read about attention-dropout