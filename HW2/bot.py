"""
HW2 — Simple bot for evaluating lab works using free local Hugging Face models.
Author: Абдулразак Зайнаб Шеху | Group: 4370 | Date: Oct 2025
"""

from transformers import pipeline

# Three free local models to compare (downloaded automatically on first run)
MODELS = [
    ("distilgpt2", 0.7, "You are a fair teacher evaluating a student’s lab work."),
    ("microsoft/phi-1_5", 0.7, "You are a strict but constructive teacher."),
    ("EleutherAI/gpt-neo-125M", 0.7, "You are a friendly teacher giving feedback."),
]

LAB_CONTENT = "Python lab: calculates the average of a list of numbers."

PROMPT_TEMPLATE = """
Analyze this student's lab report and give a detailed evaluation.

CRITERIA (max 100 points):
1. Code quality and correctness (0–30)
2. Completeness of implementation (0–30)
3. Documentation and comments (0–20)
4. Structure and formatting (0–20)

LAB CONTENT:
{lab}

Please describe:
- What the lab does
- Score each criterion with justification
- Strengths
- Improvement suggestions
- Final score out of 100
"""

def run_model(model_name: str, temperature: float, role: str) -> str:
    prompt = role + "\n\n" + PROMPT_TEMPLATE.format(lab=LAB_CONTENT)
    gen = pipeline("text-generation", model=model_name)
    out = gen(prompt, max_new_tokens=250, temperature=temperature)[0]["generated_text"]
    return out

def main():
    print(" HW2 — Testing Local Models (no tokens required)\n")
    for model_name, temp, role in MODELS:
        print("=" * 90)
        print(f"MODEL: {model_name} | temperature={temp}")
        print("=" * 90)
        try:
            result = run_model(model_name, temp, role)
            print(result[:1000])  # show first 1000 chars
        except Exception as e:
            print(f" Model error: {e}")
        print("\n")

if __name__ == "__main__":
    main()
