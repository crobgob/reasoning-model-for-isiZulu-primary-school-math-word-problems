import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, get_peft_model
import re
import numpy as np
import time
import os
from typing import List, Dict, Tuple, Optional
import logging
from torch.utils.data import DataLoader, Dataset
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleAnswerExtractor:
    """Extract answers from #### format"""
    
    def extract_answer(self, text: str) -> Optional[float]:
        """Extract numerical answer from #### format"""
        # Look for #### followed by a number
        pattern = r'####\s*([+-]?\d+(?:\.\d+)?)'
        matches = re.findall(pattern, text.strip())
        
        if matches:
            try:
                return float(matches[-1])  # Take the last match
            except ValueError:
                pass
        
        # Fallback: look for any #### followed by text containing numbers
        fallback_pattern = r'####\s*([^#\n]*)'
        fallback_matches = re.findall(fallback_pattern, text.strip())
        
        if fallback_matches:
            answer_text = fallback_matches[-1].strip()
            # Extract first number from the answer text
            number_match = re.search(r'([+-]?\d+(?:\.\d+)?)', answer_text)
            if number_match:
                try:
                    return float(number_match.group(1))
                except ValueError:
                    pass
        
        return None

class OutcomeRewardCalculator:
    """Calculate rewards based on answer correctness"""
    
    def __init__(self, tolerance=1e-6):
        self.tolerance = tolerance
        self.extractor = SimpleAnswerExtractor()
    
    def calculate_reward(self, generated_text: str, ground_truth: float) -> float:
        """Calculate reward: +1 for correct, -0.5 for wrong, -1 for no answer"""
        predicted = self.extractor.extract_answer(generated_text)
        
        if predicted is None:
            return -1.0  # Strong penalty for no extractable answer
        
        if abs(predicted - ground_truth) <= self.tolerance:
            return 1.0   # Perfect match
        elif abs(predicted - ground_truth) <= abs(ground_truth) * 0.05:  # Within 5%
            return -0.2   # Close answer
        elif abs(predicted - ground_truth) <= abs(ground_truth) * 0.1:   # Within 10%
            return -0.1   # Somewhat close
        else:
            return -0.5  # Wrong answer

class MathRLDataset(Dataset):
    """Dataset for RL training"""
    
    def __init__(self, dataframe, max_samples=None):
        self.df = dataframe.copy()
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=42)
        
        # Filter out rows with non-numeric answers
        self.df = self.df[pd.to_numeric(self.df['answer'], errors='coerce').notna()]
        self.df['answer'] = pd.to_numeric(self.df['answer'])
        
        logger.info(f"Dataset loaded with {len(self.df)} valid samples")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'question': str(row['question']).strip(),
            'ground_truth': float(row['answer']),
            'reference_cot': str(row['CoT']).strip()
        }

class REINFORCETrainer:
    """REINFORCE trainer optimized for L40 GPU"""
    
    def __init__(self, model, tokenizer, reward_calculator, learning_rate=5e-6):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_calc = reward_calculator
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        
        # Generation parameters
        self.gen_kwargs = {
            'max_new_tokens': 512,
            'do_sample': True,
            'temperature': 0.8,
            'top_p': 0.9,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        
        # Training stats
        self.stats = {
            'rewards': [],
            'losses': [],
            'accuracies': [],
            'baseline_rewards': []
        }
    
    def format_prompt(self, question: str) -> str:
        """Format question as prompt"""
        return f"Sombulula le nkinga yezibalo:\n\nQ: {question}\nA: "
    
    def generate_with_logprobs(self, prompt: str, num_samples: int = 1) -> List[Dict]:
        """Generate text and calculate log probabilities"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.model.device)
        input_length = inputs['input_ids'].shape[1]
        
        results = []
        
        for _ in range(num_samples):
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    **self.gen_kwargs
                )
            
            generated_sequence = outputs.sequences[0]
            generated_text = self.tokenizer.decode(
                generated_sequence[input_length:], 
                skip_special_tokens=True
            )
            
            # Calculate log probabilities for the generated tokens
            if hasattr(outputs, 'scores') and outputs.scores:
                log_probs = []
                for i, score in enumerate(outputs.scores):
                    if input_length + i < len(generated_sequence):
                        token_id = generated_sequence[input_length + i]
                        token_log_prob = F.log_softmax(score[0], dim=-1)[token_id]
                        log_probs.append(token_log_prob)
                
                total_log_prob = torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0)
            else:
                total_log_prob = torch.tensor(0.0)
            
            results.append({
                'text': generated_text,
                'log_prob': total_log_prob,
                'sequence': generated_sequence
            })
        
        return results
    
    def get_baseline_reward(self, prompt: str, ground_truth: float) -> float:
        """Get baseline reward using greedy generation"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            baseline_output = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,  # Greedy
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        baseline_text = self.tokenizer.decode(
            baseline_output[0][len(inputs['input_ids'][0]):],
            skip_special_tokens=True
        )
        
        return self.reward_calc.calculate_reward(baseline_text, ground_truth)
    
    def train_step(self, batch: List[Dict]) -> Dict:
        """Single training step using REINFORCE"""
        total_loss = 0.0
        total_reward = 0.0
        total_baseline_reward = 0.0
        correct_predictions = 0
        
        self.optimizer.zero_grad()
        
        for item in batch:
            question = item['question']
            ground_truth = item['ground_truth']
            prompt = self.format_prompt(question)
            
            # Generate response with log probabilities
            generations = self.generate_with_logprobs(prompt, num_samples=1)
            
            if not generations:
                continue
                
            gen = generations[0]
            generated_text = gen['text']
            log_prob = gen['log_prob']
            
            # Calculate reward
            reward = self.reward_calc.calculate_reward(generated_text, ground_truth)
            
            # Get baseline for variance reduction
            baseline_reward = self.get_baseline_reward(prompt, ground_truth)
            
            # Calculate advantage
            advantage = reward - baseline_reward
            
            # REINFORCE loss: -advantage * log_prob
            if log_prob.requires_grad:
                loss = -advantage * log_prob
                loss.backward()
                total_loss += loss.item()
            
            # Statistics
            total_reward += reward
            total_baseline_reward += baseline_reward
            if reward > 0.5:  # Consider > 0.5 as correct
                correct_predictions += 1
            
            # Log sample
            if len(batch) <= 4:  # Only log for small batches
                logger.info(f"Q: {question[:100]}...")
                logger.info(f"A: {generated_text[:150]}...")
                logger.info(f"Reward: {reward:.3f}, Baseline: {baseline_reward:.3f}, Advantage: {advantage:.3f}")
        
        # Update model
        if total_loss > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 
                max_norm=1.0
            )
            self.optimizer.step()
        
        # Calculate averages
        batch_size = len(batch)
        avg_loss = total_loss / batch_size if batch_size > 0 else 0
        avg_reward = total_reward / batch_size if batch_size > 0 else 0
        avg_baseline = total_baseline_reward / batch_size if batch_size > 0 else 0
        accuracy = correct_predictions / batch_size if batch_size > 0 else 0
        
        return {
            'loss': avg_loss,
            'reward': avg_reward,
            'baseline_reward': avg_baseline,
            'accuracy': accuracy,
            'advantage': avg_reward - avg_baseline
        }
    
    def evaluate(self, eval_dataset: Dataset, num_samples: int = 100) -> Dict:
        """Evaluate model performance"""
        logger.info("Running evaluation...")
        
        eval_indices = np.random.choice(len(eval_dataset), min(num_samples, len(eval_dataset)), replace=False)
        
        total_reward = 0.0
        correct = 0
        
        for idx in eval_indices:
            item = eval_dataset[idx]
            prompt = self.format_prompt(item['question'])
            
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs['input_ids'][0]):],
                skip_special_tokens=True
            )
            
            reward = self.reward_calc.calculate_reward(generated_text, item['ground_truth'])
            total_reward += reward
            
            if reward > 0.5:
                correct += 1
        
        accuracy = correct / len(eval_indices)
        avg_reward = total_reward / len(eval_indices)
        
        logger.info(f"Evaluation complete: Accuracy = {accuracy:.3f}, Avg Reward = {avg_reward:.3f}")
        
        return {'accuracy': accuracy, 'avg_reward': avg_reward}

def main():
    """Main training function"""
    # Set random seeds
    set_seed(42)
    
    # Configuration
    config = {
        'model_path': 'output/final_lora',
        'data_path': 'GSM8K_Zulu.parquet',
        'output_dir': 'output/rl_model',
        'learning_rate': 5e-6,
        'batch_size': 4,
        'num_epochs': 3,
        'eval_every': 100,  # Evaluate every N steps
        'save_every': 200,  # Save every N steps
        'max_train_samples': 9000,  # Limit for faster training
        'max_eval_samples': 1500,
    }
    
    logger.info("Starting REINFORCE training")
    logger.info(f"Config: {config}")
    
    # Check GPU
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("No GPU available, using CPU")
    
    # Load data
    logger.info("Loading data...")
    df = pd.read_parquet(config['data_path'])
    
    # Split data
    train_df = df.sample(frac=0.8, random_state=42)
    eval_df = df.drop(train_df.index)
    
    train_dataset = MathRLDataset(train_df, max_samples=config['max_train_samples'])
    eval_dataset = MathRLDataset(eval_df, max_samples=config['max_eval_samples'])
    
    # Load model and tokenizer  
    logger.info("Loading model...")
    base_model = "Qwen/Qwen2.5-Math-7B"  # Or whatever you fine-tuned

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, config['model_path'])  # uses LoRA adapter

    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Initialize reward calculator and trainer
    reward_calc = OutcomeRewardCalculator()
    trainer = REINFORCETrainer(model, tokenizer, reward_calc, config['learning_rate'])
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    step = 0
    best_accuracy = 0.0
    
    for epoch in range(config['num_epochs']):
        logger.info(f"📚 Epoch {epoch + 1}/{config['num_epochs']}")
        
        # Shuffle dataset
        indices = np.random.permutation(len(train_dataset))
        
        epoch_rewards = []
        epoch_losses = []
        
        for i in range(0, len(indices), config['batch_size']):
            batch_indices = indices[i:i + config['batch_size']]
            batch = [train_dataset[idx] for idx in batch_indices]
            
            # Training step
            step_results = trainer.train_step(batch)
            step += 1
            
            epoch_rewards.append(step_results['reward'])
            epoch_losses.append(step_results['loss'])
            
            # Logging
            if step % 10 == 0:
                logger.info(
                    f"Step {step}: Loss={step_results['loss']:.4f}, "
                    f"Reward={step_results['reward']:.3f}, "
                    f"Accuracy={step_results['accuracy']:.3f}, "
                    f"Advantage={step_results['advantage']:.3f}"
                )
            
            # Evaluation
            if step % config['eval_every'] == 0:
                eval_results = trainer.evaluate(eval_dataset)
                
                if eval_results['accuracy'] > best_accuracy:
                    best_accuracy = eval_results['accuracy']
                    logger.info(f"🎉 New best accuracy: {best_accuracy:.3f}")
                    
                    # Save best model
                    best_model_path = os.path.join(config['output_dir'], 'best_model')
                    os.makedirs(best_model_path, exist_ok=True)
                    model.save_pretrained(best_model_path)
                    tokenizer.save_pretrained(best_model_path)
            
            # Save checkpoint
            if step % config['save_every'] == 0:
                checkpoint_path = os.path.join(config['output_dir'], f'checkpoint-{step}')
                os.makedirs(checkpoint_path, exist_ok=True)
                model.save_pretrained(checkpoint_path)
                
                # Save training stats
                stats_path = os.path.join(checkpoint_path, 'training_stats.json')
                with open(stats_path, 'w') as f:
                    json.dump({
                        'step': step,
                        'epoch': epoch,
                        'best_accuracy': best_accuracy,
                        'recent_rewards': epoch_rewards[-10:],
                        'recent_losses': epoch_losses[-10:],
                    }, f, indent=2)
        
        # End of epoch summary
        avg_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        
        logger.info(
            f"Epoch {epoch + 1} Summary: "
            f"Avg Reward={avg_reward:.3f}, Avg Loss={avg_loss:.4f}"
        )
    
    # Final evaluation
    logger.info("Final evaluation...")
    final_results = trainer.evaluate(eval_dataset, num_samples=config['max_eval_samples'])
    
    # Save final model
    final_model_path = os.path.join(config['output_dir'], 'final_model')
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    # Training summary
    total_time = time.time() - start_time
    logger.info("Training Complete!")
    logger.info(f"⏱Total time: {total_time/3600:.2f} hours")
    logger.info(f"Best accuracy: {best_accuracy:.3f}")
    logger.info(f"Final accuracy: {final_results['accuracy']:.3f}")
    logger.info(f"Models saved to: {config['output_dir']}")

if __name__ == "__main__":
    main()