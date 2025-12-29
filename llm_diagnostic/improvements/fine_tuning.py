"""
Fine-tuning improvement strategy using LoRA/PEFT.
Implements parameter-efficient fine-tuning for task-specific adaptation.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

from .base_strategy import BaseImprovementStrategy, ImprovementConfig, ImprovementResult
from ..core.evaluator import EvaluationMetrics
from ..failure_tests.base_test import TestCase, TestResult


class FineTuningStrategy(BaseImprovementStrategy):
    """
    Fine-tune models using LoRA (Low-Rank Adaptation).
    
    More expensive than prompting but provides significant improvements
    for domain-specific tasks.
    """
    
    def __init__(self):
        super().__init__(
            name="Fine-Tuning (LoRA)",
            description="Fine-tune model on task-specific data using LoRA"
        )
        self.model = None
        self.tokenizer = None
    
    def configure(
        self,
        base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        lora_r: int = 8,
        lora_alpha: int = 32,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        **kwargs
    ) -> ImprovementConfig:
        """
        Configure fine-tuning parameters.
        
        Args:
            base_model: Base model to fine-tune
            lora_r: LoRA rank (lower = faster, less expressive)
            lora_alpha: LoRA alpha (scaling factor)
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        """
        # Estimate cost (rough approximation)
        # Assume 1000 samples, 3 epochs, ~$0.50/GPU-hour, 1 hour training
        estimated_cost = 0.50 * num_epochs
        
        return ImprovementConfig(
            strategy_name=self.name,
            parameters={
                "base_model": base_model,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": kwargs.get("batch_size", 4),
                "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 4)
            },
            estimated_cost=estimated_cost,
            estimated_time=f"{num_epochs * 30} minutes (on GPU)"
        )
    
    def apply(
        self,
        test_cases: List[TestCase],
        baseline_results: List[TestResult],
        config: ImprovementConfig,
        output_dir: str = "./fine_tuned_model",
        **kwargs
    ) -> List[TestResult]:
        """
        Fine-tune model and evaluate on test cases.
        
        Args:
            test_cases: Test cases for evaluation (NOT used for training)
            baseline_results: Baseline results
            config: Fine-tuning configuration
            output_dir: Where to save fine-tuned model
            **kwargs: Additional parameters (training_data required)
        """
        start_time = time.time()
        
        # Get training data (separate from test cases!)
        training_data = kwargs.get("training_data")
        if not training_data:
            raise ValueError("training_data required for fine-tuning")
        
        # Step 1: Load base model
        print(f"Loading base model: {config.parameters['base_model']}")
        self._load_model(config.parameters["base_model"])
        
        # Step 2: Prepare LoRA config
        lora_config = LoraConfig(
            r=config.parameters["lora_r"],
            lora_alpha=config.parameters["lora_alpha"],
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Step 3: Apply LoRA
        print("Applying LoRA...")
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Step 4: Prepare training data
        print("Preparing training data...")
        train_dataset = self._prepare_dataset(training_data)
        
        # Step 5: Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=config.parameters["num_epochs"],
            per_device_train_batch_size=config.parameters["batch_size"],
            gradient_accumulation_steps=config.parameters["gradient_accumulation_steps"],
            learning_rate=config.parameters["learning_rate"],
            logging_steps=10,
            save_strategy="epoch",
            fp16=torch.cuda.is_available(),
            optim="adamw_torch"
        )
        
        # Step 6: Train
        print("Starting training...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )
        
        trainer.train()
        
        # Step 7: Save model
        print(f"Saving fine-tuned model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Step 8: Evaluate on test cases
        print("Evaluating fine-tuned model...")
        improved_results = self._evaluate_on_test_cases(test_cases)
        
        elapsed = time.time() - start_time
        
        # Calculate metrics
        baseline_metrics = self._calculate_metrics(baseline_results)
        improved_metrics = self._calculate_metrics(improved_results)
        
        # Store result
        result = ImprovementResult(
            strategy_name=self.name,
            config=config,
            baseline_metrics=baseline_metrics,
            improved_metrics=improved_metrics,
            improvement_delta=self.evaluate_improvement(baseline_metrics, improved_metrics),
            total_cost=config.estimated_cost,  # Approximate
            total_time_seconds=elapsed,
            metadata={
                "output_dir": output_dir,
                "trainable_params": self.model.get_nb_trainable_parameters()
            }
        )
        
        self.results.append(result)
        return improved_results
    
    def _load_model(self, model_name: str):
        """Load base model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add pad token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with 4-bit quantization for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            load_in_4bit=torch.cuda.is_available()  # 4-bit quantization
        )
        
        # Prepare for training
        if torch.cuda.is_available():
            self.model = prepare_model_for_kbit_training(self.model)
    
    def _prepare_dataset(self, training_data: List[Dict[str, str]]) -> Dataset:
        """
        Prepare dataset for training.
        
        Expected format: [{"input": "...", "output": "..."}, ...]
        """
        # Format as instruction-following
        formatted_data = []
        for item in training_data:
            text = f"### Input:\n{item['input']}\n\n### Output:\n{item['output']}"
            formatted_data.append({"text": text})
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        dataset = Dataset.from_list(formatted_data)
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        return tokenized_dataset
    
    def _evaluate_on_test_cases(self, test_cases: List[TestCase]) -> List[TestResult]:
        """Evaluate fine-tuned model on test cases."""
        results = []
        
        self.model.eval()
        
        for test_case in test_cases:
            # Prepare input
            input_text = f"### Input:\n{test_case.input}\n\n### Output:\n"
            inputs = self.tokenizer(input_text, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode
            prediction = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Evaluate
            success = test_case.expected_output.lower() in prediction.lower() if test_case.expected_output else True
            
            results.append(TestResult(
                test_case_id=test_case.id,
                prediction=prediction,
                reference=test_case.expected_output,
                success=success,
                metrics={"accuracy": 1.0 if success else 0.0},
                response=None  # No cost tracking for local inference
            ))
        
        return results
    
    def _calculate_metrics(self, results: List[TestResult]) -> EvaluationMetrics:
        """Calculate aggregate metrics."""
        metrics = EvaluationMetrics()
        
        if not results:
            return metrics
        
        accuracy = sum(r.metrics.get("accuracy", 0) for r in results) / len(results)
        metrics.add_metric("accuracy", accuracy)
        
        return metrics
    
    def get_implementation_guide(self) -> Dict[str, Any]:
        """Implementation guide for fine-tuning."""
        return {
            "overview": "Fine-tune models using LoRA for task-specific improvements",
            "when_to_use": [
                "Task requires domain-specific knowledge",
                "Prompt engineering plateaus at ~80% accuracy",
                "Have 500+ labeled examples",
                "Need consistent formatting or style",
                "Want to reduce inference cost (smaller model)"
            ],
            "requirements": {
                "data": "500-5000 labeled examples (more is better)",
                "compute": "GPU with 16GB+ VRAM (or use Colab/Modal)",
                "time": "1-4 hours depending on dataset size",
                "cost": "$0.50-$5 for training"
            },
            "expected_improvements": {
                "domain_specific": "15-30% accuracy gain",
                "style_consistency": "40-60% improvement",
                "cost_reduction": "10-100x cheaper than GPT-4"
            },
            "best_practices": [
                "Start with 500-1000 examples, add more if needed",
                "Use LoRA (not full fine-tuning) for efficiency",
                "Keep base model frozen, only tune adapters",
                "Validate on held-out test set",
                "Monitor for overfitting (train vs val accuracy)",
                "Use smaller models (7B) if possible for cost"
            ],
            "pitfalls": [
                "Overfitting on small datasets",
                "Catastrophic forgetting (losing general capabilities)",
                "Need to retrain when data distribution changes",
                "Higher upfront cost than prompting"
            ],
            "code_example": """
from llm_diagnostic.improvements import FineTuningStrategy

# Prepare training data
training_data = [
    {"input": "Extract entities from: John works at Google", 
     "output": "Person: John, Organization: Google"},
    # ... more examples
]

# Configure
strategy = FineTuningStrategy()
config = strategy.configure(
    base_model="mistralai/Mistral-7B-Instruct-v0.2",
    lora_r=8,
    num_epochs=3
)

# Fine-tune
improved_results = strategy.apply(
    test_cases=test_cases,
    baseline_results=baseline_results,
    config=config,
    training_data=training_data,
    output_dir="./my_fine_tuned_model"
)
"""
        }


# Example usage
if __name__ == "__main__":
    from ..failure_tests.base_test import TestCase
    
    # Example training data
    training_data = [
        {"input": "What is 2+2?", "output": "4"},
        {"input": "What is 5+7?", "output": "12"},
        # ... more examples
    ]
    
    # Test cases (separate from training!)
    test_cases = [
        TestCase(id="test1", input="What is 3+4?", expected_output="7")
    ]
    
    strategy = FineTuningStrategy()
    config = strategy.configure(num_epochs=1)  # Quick test
    
    # Would run: strategy.apply(test_cases, [], config, training_data=training_data)
    print("Fine-tuning strategy configured")
    print(strategy.get_implementation_guide())