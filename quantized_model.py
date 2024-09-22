import transformers
import torch


def load_text_generation_pipeline(model_id="unsloth/llama-3-8b-Instruct-bnb-4bit"):
    # Load the text generation pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.float16,
            "quantization_config": {"load_in_4bit": False},
            "device_map": "auto",
            "low_cpu_mem_usage": True,
        },
    )
    return pipeline


# 1D ARC
# ground truths
# prompting
#


def generate_response(pipeline, system_prompt, user_query, max_new_tokens=256):
    # Use the pipeline to generate a response
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=max_new_tokens,
    )
    # Return the generated text
    return outputs[0]["generated_text"]


# Example usage:
if __name__ == "__main__":
    # Load the pipeline once
    pipeline = load_text_generation_pipeline()

    # Generate a response
    response = generate_response(
        pipeline,
        system_prompt="You are a pirate chatbot who always responds in pirate speak!",
        user_query="Who are you?",
    )

    # Print the response
    print(response)
