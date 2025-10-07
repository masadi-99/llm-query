import os
import json
import argparse
from pathlib import Path
from typing import List, Dict
from vllm import LLM, SamplingParams

def load_reports(reports_dir: str) -> Dict[str, str]:
    """Load all report files from the specified directory."""
    reports = {}
    reports_path = Path(reports_dir)
    
    if not reports_path.exists():
        raise ValueError(f"Reports directory not found: {reports_dir}")
    
    for file_path in reports_path.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as f:
            reports[file_path.stem] = f.read()
    
    print(f"Loaded {len(reports)} reports")
    return reports

def create_prompts(reports: Dict[str, str], question: str) -> tuple[List[str], List[str]]:
    """Create prompts for each report with the given question."""
    prompts = []
    report_names = []
    
    for report_name, report_content in reports.items():
        prompt = f"""Based on the following report, please answer this question: {question}

Report:
{report_content}

Answer:"""
        prompts.append(prompt)
        report_names.append(report_name)
    
    return prompts, report_names

def main():
    parser = argparse.ArgumentParser(description='Query reports using vLLM')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name or path (e.g., meta-llama/Llama-2-7b-chat-hf)')
    parser.add_argument('--reports-dir', type=str, required=True,
                       help='Directory containing report files')
    parser.add_argument('--question', type=str, 
                       default="What are the key findings in this report?",
                       help='Question to ask about each report')
    parser.add_argument('--output-file', type=str, default='~/report_results.json',
                       help='Output file for results')
    parser.add_argument('--max-model-len', type=int, default=4096,
                       help='Maximum context length for the model')
    parser.add_argument('--max-tokens', type=int, default=512,
                       help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Sampling temperature')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                       help='Number of GPUs to use for tensor parallelism')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9,
                       help='GPU memory utilization (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Expand home directory in output path
    output_file = Path(args.output_file).expanduser()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {args.model}")
    print(f"Max context length: {args.max_model_len}")
    print(f"Max output tokens: {args.max_tokens}")
    
    # Initialize vLLM
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True
    )
    
    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=0.95
    )
    
    # Load reports and create prompts
    print(f"\nLoading reports from: {args.reports_dir}")
    reports = load_reports(args.reports_dir)
    
    print(f"\nQuestion: {args.question}")
    prompts, report_names = create_prompts(reports, args.question)
    
    print(f"\nProcessing {len(prompts)} prompts in batch...")
    
    # Generate responses for all prompts in one batch
    outputs = llm.generate(prompts, sampling_params)
    
    # Collect results
    results = []
    for report_name, output in zip(report_names, outputs):
        result = {
            'report_name': report_name,
            'question': args.question,
            'answer': output.outputs[0].text.strip(),
            'prompt_tokens': len(output.prompt_token_ids),
            'completion_tokens': len(output.outputs[0].token_ids)
        }
        results.append(result)
        print(f"\nReport: {report_name}")
        print(f"Answer: {result['answer'][:200]}...")
    
    # Save results
    print(f"\nSaving results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\nProcessing complete!")
    print(f"Total reports processed: {len(results)}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()
