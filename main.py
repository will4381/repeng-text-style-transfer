import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, BitsAndBytesConfig
from tqdm import tqdm
import gc
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu

nltk.download('punkt', quiet=True)

class DatasetEntry:
    def __init__(self, positive, negative):
        self.positive = positive
        self.negative = negative

# Repeng library modifications
class AdvancedControlVector:
    def __init__(self, model_type, directions, sbert_model, lrp_weights, model, tokenizer, style_dataset):
        self.model_type = model_type
        self.directions = directions
        self.sbert_model = sbert_model
        self.lrp_weights = lrp_weights
        self.style_directions = {}
        self.vocabulary_directions = {}
        self.structure_directions = {}
        self.model = model
        self.tokenizer = tokenizer
        self.style_dataset = style_dataset

    @classmethod
    def _train_base(cls, model, tokenizer, dataset, sbert_model, lrp_weights, batch_size=32, **kwargs):
        all_hidden_states = []
        max_length = max(len(tokenizer.encode(entry.positive)) for entry in dataset)

        for i in tqdm(range(0, len(dataset), batch_size), desc="Computing hidden states"):
            batch = dataset[i:i+batch_size]
            inputs = tokenizer([entry.positive for entry in batch], return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
            hidden_states = torch.stack(outputs.hidden_states).squeeze()
            all_hidden_states.append(hidden_states)

        all_hidden_states = torch.cat(all_hidden_states, dim=1)

        directions = {}
        for layer in range(0, model.config.num_hidden_layers, 8):
            layer_states = all_hidden_states[layer]
            layer_states_2d = layer_states.view(-1, layer_states.size(-1)).cpu().numpy()
            pca = PCA(n_components=1)
            pca.fit(layer_states_2d)
            direction = pca.components_[0]
            directions[layer] = direction

        return cls(model.config.model_type, directions, sbert_model, lrp_weights, model, tokenizer, dataset)

    @classmethod
    def train(cls, model, tokenizer, dataset, sbert_model, **kwargs):
        lrp_weights = cls._compute_lrp_weights(model, tokenizer, dataset)
        base_vector = cls._train_base(model, tokenizer, dataset, sbert_model, lrp_weights, **kwargs)
        base_vector._train_specialized_directions(model, tokenizer, dataset)
        return base_vector

    @classmethod
    def _compute_lrp_weights(cls, model, tokenizer, dataset):
        lrp_weights = {}
        subset_size = min(10, len(dataset))
        subset = random.sample(dataset, subset_size)

        for entry in tqdm(subset, desc="Computing LRP weights"):
            input_ids = tokenizer.encode(entry.positive, return_tensors="pt").to(model.device)
            for layer in range(0, model.config.num_hidden_layers, 2):
                if layer not in lrp_weights:
                    lrp_weights[layer] = []
                try:
                    lrp_result = memory_efficient_lrp(model, input_ids, layer)
                    lrp_weights[layer].append(lrp_result)
                except Exception as e:
                    print(f"Error computing LRP for layer {layer}: {str(e)}")
                    continue
                torch.cuda.empty_cache()

        averaged_weights = {layer: np.mean(weights, axis=0) for layer, weights in lrp_weights.items() if weights}
        return averaged_weights

    def _train_specialized_directions(self, model, tokenizer, dataset):
        style_directions = {}
        vocab_directions = {}
        structure_directions = {}

        max_length = max(len(tokenizer.encode(entry.positive)) for entry in dataset)

        for entry in tqdm(dataset, desc="Training specialized directions"):
            inputs = tokenizer(entry.positive, return_tensors="pt", padding="max_length", truncation=True, max_length=max_length).to(model.device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

            hidden_states = torch.stack(outputs.hidden_states).squeeze()

            # Capture style information from multiple layers
            style_direction = hidden_states[-3:].mean(dim=0).mean(dim=1).detach().cpu().numpy()

            token_embeddings = model.get_input_embeddings()(inputs.input_ids)
            vocab_direction = token_embeddings.mean(dim=1).detach().cpu().numpy()

            # Capture structural information
            structure_direction = outputs.attentions[-1].mean(dim=(0, 1)).detach().cpu().numpy()

            for layer in range(model.config.num_hidden_layers):
                if layer not in style_directions:
                    style_directions[layer] = []
                    vocab_directions[layer] = []
                    structure_directions[layer] = []

                style_directions[layer].append(style_direction)
                vocab_directions[layer].append(vocab_direction)
                structure_directions[layer].append(structure_direction)

        for layer in range(model.config.num_hidden_layers):
            self.style_directions[layer] = np.mean(style_directions[layer], axis=0)
            self.vocabulary_directions[layer] = np.mean(vocab_directions[layer], axis=0)
            self.structure_directions[layer] = np.mean(structure_directions[layer], axis=0)

# Layer-wise Relevance Propagation (LRP) implementation
def memory_efficient_lrp(model, input_ids, target_layer):
    model.eval()
    input_embed = model.model.embed_tokens(input_ids).requires_grad_(True)

    target_hidden_state = None
    def forward_hook(module, input, output):
        nonlocal target_hidden_state
        target_hidden_state = output[0] if isinstance(output, tuple) else output

    hook = model.model.layers[target_layer].register_forward_hook(forward_hook)

    outputs = model(inputs_embeds=input_embed)
    last_token_logits = outputs.logits[:, -1, :]

    hook.remove()

    if target_hidden_state is None:
        raise ValueError(f"Failed to capture hidden state for layer {target_layer}")

    relevance = torch.zeros_like(target_hidden_state)
    for i in range(target_hidden_state.size(1)):
        model.zero_grad()
        grad = torch.autograd.grad(last_token_logits.sum(), target_hidden_state, retain_graph=True)[0]
        relevance[:, i, :] = target_hidden_state[:, i, :] * grad[:, i, :]

    relevance = relevance / relevance.sum()

    return relevance.mean(dim=1).detach().cpu().numpy()

class DynamicControlModel(torch.nn.Module):
    def __init__(self, model: PreTrainedModel, layer_ids, style_checker):
        super().__init__()
        self.model = model
        self.layer_ids = layer_ids
        self.style_checker = style_checker
        self.control_strength = 2.0
        self.raw_control = None

    def set_raw_control(self, control):
        self.raw_control = control

    def generate(self, input_ids, max_length, temperature, **kwargs):
        return self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            **kwargs
        )

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        if self.raw_control is not None and hasattr(outputs, 'hidden_states'):
            hidden_states = list(outputs.hidden_states)
            for layer_id, control in self.raw_control.items():
                if layer_id < len(hidden_states):
                    hidden_states[layer_id] = hidden_states[layer_id] + self.control_strength * control.to(hidden_states[layer_id].device)
            outputs.hidden_states = tuple(hidden_states)
        return outputs

    def _adjust_control_strength(self, style_score):
        self.control_strength = 1.0 + (1.0 - style_score)

class IterativeControlVector(AdvancedControlVector):
    def __init__(self, model_type, directions, sbert_model, lrp_weights, model, tokenizer, style_dataset):
        super().__init__(model_type, directions, sbert_model, lrp_weights, model, tokenizer, style_dataset)
        self.style_embeddings = self._compute_style_embeddings()
        self.projection = None  # We initialize this in contrastive_loss

    def _compute_style_embeddings(self):
        style_sentences = [entry.positive for entry in self.style_dataset]
        return self.sbert_model.encode(style_sentences, show_progress_bar=True)

    def style_similarity(self, text):
        text_embedding = self.sbert_model.encode([text], convert_to_tensor=True).to(self.model.device)
        model_embedding = self.model(**self.tokenizer(text, return_tensors="pt").to(self.model.device), output_hidden_states=True).hidden_states[-1][:, 0, :]
        projected_embedding = self.projection(model_embedding.float())
        similarities = F.cosine_similarity(projected_embedding, text_embedding.float())
        return similarities.mean().item()

    @classmethod
    def train(cls, model, tokenizer, dataset, contrastive_dataset, sbert_model, lrp_weights, 
          num_iterations=20, early_stop_threshold=0.0001, batch_size=8, accumulation_steps=4, **kwargs):
        augmented_dataset = cls._augment_data(dataset)
        base_vector = cls._train_base(model, tokenizer, augmented_dataset, sbert_model, lrp_weights, **kwargs)
        base_vector._train_specialized_directions(model, tokenizer, augmented_dataset)
        advanced_vector = cls(base_vector.model_type, base_vector.directions, sbert_model, lrp_weights, model, tokenizer, dataset)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        
        prev_similarity = float('-inf')
        for iteration in range(num_iterations):
            total_loss = 0
            total_similarity = 0
            
            for i in tqdm(range(0, len(augmented_dataset), batch_size), desc=f"Iteration {iteration + 1}"):
                batch = augmented_dataset[i:i+batch_size]
                
                input_texts = [entry.positive for entry in batch]
                input_ids = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
                
                outputs = model(**input_ids, output_hidden_states=True)
                generated_embeddings = outputs.hidden_states[-1][:, 0, :]
                
                style_embeddings = sbert_model.encode(input_texts, convert_to_tensor=True).to(model.device)
                
                contrastive_loss = advanced_vector.contrastive_loss(generated_embeddings, style_embeddings)
                
                loss = contrastive_loss / accumulation_steps
            
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"Warning: NaN or Inf loss detected. Skipping this batch.")
                    continue
                
                loss.backward()
                
                if (i + batch_size) % (batch_size * accumulation_steps) == 0 or (i + batch_size) >= len(augmented_dataset):
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps
                
                with torch.no_grad():
                    generated_embeddings_projected = advanced_vector.projection(generated_embeddings.float())
                    similarities = F.cosine_similarity(generated_embeddings_projected, style_embeddings.float())
                    if torch.isnan(similarities).any() or torch.isinf(similarities).any():
                        print(f"Warning: NaN or Inf similarity detected. Skipping this batch.")
                        continue
                    total_similarity += similarities.mean().item()
            
            avg_loss = total_loss / len(augmented_dataset)
            avg_similarity = total_similarity / (len(augmented_dataset) / batch_size)
            
            print(f"Iteration {iteration + 1}: Avg Loss = {avg_loss:.4f}, Avg Similarity = {avg_similarity:.4f}")
            
            if iteration % 5 == 0:
                advanced_vector.evaluate_and_log(iteration)
            
            if iteration > 0 and (avg_similarity - prev_similarity) < early_stop_threshold:
                print(f"Early stopping at iteration {iteration + 1}")
                break
            
            prev_similarity = avg_similarity
        
        return advanced_vector

    def contrastive_loss(self, generated_embeddings, style_embeddings, temperature=0.07):
        if self.projection is None or self.projection.in_features != generated_embeddings.size(-1) or self.projection.out_features != style_embeddings.size(-1):
            self.projection = nn.Linear(generated_embeddings.size(-1), style_embeddings.size(-1), dtype=torch.float32).to(generated_embeddings.device)
        
        generated_embeddings = generated_embeddings.to(torch.float32)
        style_embeddings = style_embeddings.to(torch.float32)
        
        generated_embeddings_projected = self.projection(generated_embeddings)
        
        similarity_matrix = F.cosine_similarity(generated_embeddings_projected.unsqueeze(1), style_embeddings.unsqueeze(0), dim=2)
        positive_pairs = torch.diag(similarity_matrix)
        negative_pairs = similarity_matrix - torch.eye(similarity_matrix.size(0)).to(self.model.device)
        
        positive_loss = -torch.log(torch.exp(positive_pairs / temperature) / torch.sum(torch.exp(similarity_matrix / temperature), dim=1))
        negative_loss = -torch.log(1 - torch.exp(negative_pairs / temperature) / torch.sum(torch.exp(similarity_matrix / temperature), dim=1))
        
        loss = (positive_loss + negative_loss.sum(dim=1)) / (similarity_matrix.size(0) - 1)
        return loss.mean()

    @staticmethod
    def _augment_data(dataset):
        augmented_dataset = []
        for entry in dataset:
            augmented_dataset.append(entry)
            # Simple word swap augmentation
            words = word_tokenize(entry.positive)
            if len(words) > 3:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
                augmented_entry = DatasetEntry(positive=' '.join(words), negative=entry.negative)
                augmented_dataset.append(augmented_entry)
        return augmented_dataset

    def _refine_vector(self, dataset, generated_samples, contrastive_dataset):
        original_embeddings = self.sbert_model.encode([entry.positive for entry in dataset])
        generated_embeddings = self.sbert_model.encode(generated_samples)
        contrastive_embeddings = self.sbert_model.encode([entry.positive for entry in contrastive_dataset])

        style_similarities = cosine_similarity(original_embeddings, generated_embeddings)
        contrastive_similarities = cosine_similarity(contrastive_embeddings, generated_embeddings)

        for layer, vector in self.directions.items():
            style_adjustment = np.mean(style_similarities) - 0.5
            contrastive_adjustment = 0.5 - np.mean(contrastive_similarities)

            # Incorporate specialized directions
            style_component = self.style_directions.get(layer, np.zeros_like(vector))
            vocab_component = self.vocabulary_directions.get(layer, np.zeros_like(vector))
            structure_component = self.structure_directions.get(layer, np.zeros_like(vector))

            combined_adjustment = (
                style_adjustment * 0.5 * style_component +
                style_adjustment * 0.3 * vocab_component +
                style_adjustment * 0.1 * structure_component +
                contrastive_adjustment * 0.1 * vector
            )

            self.directions[layer] += combined_adjustment * 1.5  # Increase the overall adjustment

        # Add L2 normalization
        for layer in self.directions:
            self.directions[layer] /= np.linalg.norm(self.directions[layer])

        return self

    def evaluate_style_transfer(self, original_text, styled_text):
        # Compute BLEU score
        reference = [word_tokenize(original_text.lower())]
        candidate = word_tokenize(styled_text.lower())
        bleu_score = sentence_bleu(reference, candidate)

        # Compute style similarity
        style_similarity = self.style_similarity(styled_text)

        return {
            'bleu_score': bleu_score,
            'style_similarity': style_similarity
        }

    def evaluate_and_log(self, iteration):
        prompts = [
            "What's interesting about artificial intelligence?",
            "How does communication matter in business?",
            "What do you think about social media's effects on society?"
        ]

        print(f"\nEvaluation at Iteration {iteration + 1}:")
        for prompt in prompts:
            styled_text = self.generate_styled_text(prompt)
            similarity = self.style_similarity(styled_text)
            perplexity = self.calculate_perplexity(styled_text)

            print(f"\nPrompt: {prompt}")
            print(f"Style Similarity: {similarity:.4f}")
            print(f"Perplexity: {perplexity:.2f}")
            print(f"Generated Text: {styled_text[:100]}...")

    def calculate_perplexity(self, text):
        tokens = self.tokenizer.encode(text)
        input_ids = torch.tensor([tokens]).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
        return torch.exp(loss).item()

    def generate_samples(self, dataset):
        samples = []
        for entry in tqdm(dataset, desc="Generating samples"):
            prompt = entry.positive.split(": ", 1)[-1]
            styled_text = self.generate_styled_text(prompt)
            samples.append(styled_text)
        return samples

    def generate_styled_text(self, prompt, max_length=150):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.model.device)

        lrp_adjusted_control = {}
        for layer, vector in self.directions.items():
            if layer in self.lrp_weights:
                lrp_adjusted_control[layer] = torch.tensor(vector * self.lrp_weights[layer] * 2.0).to(self.model.device)
            else:
                print(f"Warning: LRP weights not found for layer {layer}")
                lrp_adjusted_control[layer] = torch.tensor(vector * 2.0).to(self.model.device)

        control_model = DynamicControlModel(self.model, list(lrp_adjusted_control.keys()), self)
        control_model.set_raw_control(lrp_adjusted_control)

        def dynamic_temperature_fn(logits):
            return self.dynamic_temperature(logits)

        output = control_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_p=0.92,
            temperature=self.dynamic_temperature,
            repetition_penalty=1.2,
            pad_token_id=self.tokenizer.eos_token_id
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)

    def dynamic_temperature(self, logits):
        prob_distribution = F.softmax(logits, dim=-1)
        entropy = -torch.sum(prob_distribution * torch.log(prob_distribution + 1e-9), dim=-1)
        
        # Consider linguistic complexity
        top_k_probs, _ = torch.topk(prob_distribution, k=10, dim=-1)
        complexity = torch.std(top_k_probs, dim=-1)  # Higher std indicates more complex choice
        
        temperature = 0.5 + 0.3 * torch.sigmoid(entropy - 5) + 0.2 * complexity
        return temperature.mean().item()  # Return a scalar

def main():
    # File paths
    input_file_path = "/content/drive/MyDrive/data/input.jsonl"
    your_style_output_path = "/content/drive/MyDrive/data/vec_output_1.jsonl"
    contrastive_output_path = "/content/drive/MyDrive/data/contr_output_1.jsonl"

    # Load and process data
    def load_data(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {file_path}")
            return None

    your_style_data = load_data(your_style_output_path)
    contrastive_data = load_data(contrastive_output_path)

    if your_style_data is None or contrastive_data is None:
        print("Error loading data. Exiting.")
        return

    style_dataset = [
        DatasetEntry(positive=s, negative="")
        for i, s in enumerate(your_style_data['sentences']) if i % 4 == 0
    ][:200]

    contrastive_dataset = [
        DatasetEntry(positive=s, negative="")
        for s in contrastive_data['sentences']
    ][:200]

    # Load and prepare the model
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        return

    torch.cuda.empty_cache()
    gc.collect()

    sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
    sbert_model.to(torch.float32)

    model.train()

    lrp_weights = AdvancedControlVector._compute_lrp_weights(model, tokenizer, style_dataset)

    control_vector = IterativeControlVector.train(
        model, tokenizer, style_dataset, contrastive_dataset, sbert_model, lrp_weights, 
        num_iterations=20, early_stop_threshold=0.0001, batch_size=8, accumulation_steps=4
    )

    prompts = [
        "What's interesting about artificial intelligence?",
        "How does communication matter in business?",
        "What do you think about social media's effects on society?"
    ]

    print("\nFinal Generated Texts:")
    for prompt in prompts:
        full_prompt = f"[INST] Write about this topic in my personal style: {prompt} [/INST]"
        styled_text = control_vector.generate_styled_text(full_prompt)
        similarity = control_vector.style_similarity(styled_text)
        perplexity = control_vector.calculate_perplexity(styled_text)
        
        print(f"\nPrompt: {prompt}")
        print(f"Style Similarity: {similarity:.4f}")
        print(f"Perplexity: {perplexity:.2f}")
        print(styled_text)
        print("-" * 50)

    del control_vector
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
