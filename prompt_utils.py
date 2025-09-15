import json
import torch
import numpy as np
from tqdm import tqdm

def get_topk_similar_words(model, prompt, base_word, vocab, k=30):
    """
    Find top-k similar words based on text embedding similarity

    Args:
        model: Text encoder model
        prompt: Base prompt template
        base_word: Reference word for similarity comparison
        vocab: Vocabulary list to search from
        k: Number of similar words to return

    Returns:
        List of top-k similar words
    """
    # Tokenize base prompt
    text_input = model.tokenizer(
        [prompt.format(word=base_word)],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    # Get text embedding for base prompt
    with torch.no_grad():
        encoder_output = model.text_encoder(text_input.input_ids.to(model.device))
    full_prompt_embedding = encoder_output.pooler_output
    full_prompt_embedding = full_prompt_embedding / full_prompt_embedding.norm(p=2, dim=-1, keepdim=True)

    # Process vocabulary in batches
    prompts = [prompt.format(word=word) for word in vocab]
    batch_size = 1000
    all_prompts_embeddings = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        curr_prompts = prompts[i:i + batch_size]
        with torch.no_grad():
            text_input = model.tokenizer(
                curr_prompts,
                padding="max_length",
                max_length=model.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            curr_embeddings = model.text_encoder(text_input.input_ids.to(model.device)).pooler_output
        all_prompts_embeddings.append(curr_embeddings)

    # Calculate similarities
    all_prompts_embeddings = torch.cat(all_prompts_embeddings)
    all_prompts_embeddings = all_prompts_embeddings / all_prompts_embeddings.norm(p=2, dim=-1, keepdim=True)
    prompts_similarities = all_prompts_embeddings.matmul(full_prompt_embedding.view(-1, 1))
    sorted_prompts_similarities = np.flip(prompts_similarities.cpu().numpy().reshape(-1).argsort())

    # Print results
    print(f"prompt: {prompt}")
    print(f"initial word: {base_word}")
    print(f"TOP {k} SIMILAR WORDS:")
    similar_words = [vocab[index] for index in sorted_prompts_similarities[:k]]
    print(similar_words)
    return similar_words


def get_proxy_words(args, ldm_stable):
    """
    Get proxy words for adversarial attack

    Args:
        args: Command line arguments
        ldm_stable: Stable diffusion model

    Returns:
        List of proxy words including the target object
    """
    if len(args.proxy_words) > 0:
        return [args.object_of_interest] + args.proxy_words

    # Load and filter vocabulary
    vocab = list(json.load(open("vocab.json")).keys())
    vocab = [word for word in vocab if word.isalpha() and len(word) > 1]

    # Get similar words at different specificity levels
    filtered_vocab = get_topk_similar_words(
        ldm_stable,
        "a photo of a {word}",
        args.object_of_interest,
        vocab,
        k=50
    )

    proxy_words = get_topk_similar_words(
        ldm_stable,
        args.prompt,
        args.object_of_interest,
        filtered_vocab,
        k=args.number_of_variations
    )

    # Ensure base word is included
    if proxy_words[0] != args.object_of_interest:
        proxy_words = [args.object_of_interest] + proxy_words

    return proxy_words


def get_proxy_prompts(args, ldm_stable):
    """
    Generate proxy prompts for adversarial attack

    Args:
        args: Command line arguments
        ldm_stable: Stable diffusion model

    Returns:
        Tuple containing proxy words, base prompts, and proxy prompt objects
    """
    proxy_words = get_proxy_words(args, ldm_stable)
    base_prompt = args.prompt.format(word=args.object_of_interest)
    prompts = [base_prompt]
    proxy_prompts = [{"word": word, "prompt": args.prompt.format(word=word)} for word in proxy_words]

    return proxy_words, prompts, proxy_prompts