# train_grpo.py - Simple and Fixed
import re
import pandas as pd
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

def main():
    # 1. Load the Zulu GSM8K dataset
    parquet_file = 'GSM8K_Zulu.parquet'
    df = pd.read_parquet(parquet_file)
    
    # 2. Create prompts for each question
    prompts = []
    for question in df['question']:
        prompt = (
            "Xazulula le nkinga yezibalo, ubonise ukusebenza kwakho isinyathelo ngesinyathelo. "
            "Sebenzisa le fomethi kuphela: <cabanga>…</cabanga><phendula>…</phendula>.\n\n"
            f"Umbuzo: {question}"
        )
        prompts.append(prompt)
    
    # 3. Create dataset with prompts and ground truth
    ground_truth = df['answer'].astype(str).tolist()
    dataset = Dataset.from_dict({
        "prompt": prompts,
        "ground_truth": ground_truth
    })
    
    # 4. Define reward function
    def format_reward_func(completions, ground_truth, **kwargs):
        """Reward function:
        - 2 if format is correct AND answer matches ground_truth
        - 1 if format is correct only
        - 0 otherwise
        """
        rewards = []
        pattern = r"^<cabanga>.*?</cabanga><phendula>(.*?)</phendula>$"
        
        # Handle different completion formats
        completion_contents = []
        for completion in completions:
            if isinstance(completion, list) and len(completion) > 0:
                completion_contents.append(completion[0]["content"])
            elif isinstance(completion, str):
                completion_contents.append(completion)
            else:
                completion_contents.append("")
        
        for content, gt in zip(completion_contents, ground_truth):
            match = re.match(pattern, content, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                if answer == gt.strip():
                    rewards.append(2.0)  # Correct format + correct answer
                else:
                    rewards.append(1.0)  # Correct format, wrong answer
            else:
                rewards.append(0.0)      # Wrong format
        return rewards
    
    # 5. Configure LoRA (Parameter-efficient training)
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=32,                    # Rank (lower = less memory)
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    
    # 6. Configure training settings
    training_args = GRPOConfig(
        output_dir="output/Qwen2.5-Math-7B-GRPO",
        per_device_train_batch_size=1,        # Small batch size for memory
        gradient_accumulation_steps=8,        # Accumulate gradients
        bf16=True,                 # Save memory
        max_completion_length=512,            # Limit completion length
        max_prompt_length=512,                # Limit prompt length
        num_generations=8,                    # Number of generations per prompt
        remove_unused_columns=False,          # Keep ground_truth column
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
    )
    
    # 7. Create trainer
    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-Math-7B",         #Qwen/Qwen2-0.5B-Instruct; "Qwen/Qwen2.5-Math-7B"
        peft_config=peft_config,              # Add LoRA configuration
        reward_funcs=[format_reward_func],    # Fixed: use list and correct function name
        args=training_args,
        train_dataset=dataset,
    )
    
    # 8. Start training
    print(f"Starting training with {len(dataset)} examples...")
    trainer.train()
    print("Training completed!")

if __name__ == "__main__":
    main()