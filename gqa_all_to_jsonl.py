import pandas as pd

# https://huggingface.co/datasets/lmms-lab-encoder/GQA/tree/main/val_all_instructions
df = pd.read_parquet("./val-00000-of-00002.parquet")

df = df[df["isBalanced"] == False]

df_yes = df[df["answer"] == "yes"][["imageId", "question", "answer", "isBalanced"]].head(500)
df_no  = df[df["answer"] == "no" ][["imageId", "question", "answer", "isBalanced"]].head(500)
df_all = pd.concat([df_yes, df_no])

df_all.to_json("output.jsonl", orient="records", lines=True)
