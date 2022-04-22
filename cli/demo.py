import gradio as gr
import torch
from transformers import T5ForConditionalGeneration, RobertaTokenizer

max_seq_length = 128
model_dir = "txt_sql_output_dir/old"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = T5ForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_dir, local_files_only=True
)

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")


def question_to_sql(question):
    input_ids = tokenizer(
        question, max_length=max_seq_length, truncation=True, return_tensors="pt"
    ).input_ids

    pred = model.generate(
        input_ids,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        # attention_mask=mask,
        max_length=max_seq_length,
    )

    return tokenizer.decode(pred[0], skip_special_tokens=True)


iface = gr.Interface(
    fn=question_to_sql,
    inputs=gr.inputs.Textbox(lines=1, placeholder="Ask a question..."),
    outputs="text",
)

if __name__ == "__main__":
    app, local_url, share_url = iface.launch()
    # question_to_sql("hello world")

