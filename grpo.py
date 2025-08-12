# train_grpo.py
from datasets import load_dataset
import re
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
from peft import LoraConfig
from datasets import Dataset

# Load the Zulu GSM8K dataset
parquet_file = 'GSM8K_Zulu.parquet'
df = pd.read_parquet(parquet_file)

# Filter dataset to only include question and answer columns (exclude 'cot')
filtered_df = df[['question', 'answer']].copy()

# Add instruction to questions if needed
filtered_df['question'] = filtered_df['question'].apply(
    lambda q: f"Phendula le nkinga yezibalo isinyathelo ngesinyathelo usebenzisa ifomethi ethi <cabanga>ukucabanga</cabanga><phendula>impendulo</phendula>:\n{q}"
)

# Convert pandas DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(filtered_df)

# Extract ground truth answers as strings
ground_truth = filtered_df['answer'].astype(str).tolist()



def reward_func(completions, ground_truth, **kwargs):
    """Reward function:
    - 2 if format is correct AND answer matches ground_truth
    - 1 if format is correct only
    - 0 otherwise
    """
    rewards = []
    pattern = r"^<cabanga>.*?</cabanga><phendula>(.*?)</phendula>$"
    
    completion_contents = [completion[0]["content"] for completion in completions]
    
    for content, gt in zip(completion_contents, ground_truth):
        print(f"Content: {content}, Ground Truth: {gt}")
        match = re.match(pattern, content, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            if answer == gt.strip():
                rewards.append(2.0)
            else:
                rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards


def main():
    training_args = GRPOConfig(
        output_dir="Qwen2.5-Math-7B-GRPO",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        num_generations=8, 
        bf16=True,
        max_completion_length=512,            # Limit completion length
        max_prompt_length=512, 
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=False  # Needed to access ground_truth column
    )

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=32,  # Increase rank for more capacity (but more VRAM)
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # More modules
        bias="none",
    )

    trainer = GRPOTrainer(
        model="Qwen/Qwen2.5-Math-7B",
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config #Use Lora

    )

    trainer.train()

if __name__ == "__main__":
    main()

