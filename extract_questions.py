import json
import os


def prepare_finetuning_data(input_filepath, output_filepath):
    """
    Extracts Q&A pairs from the news JSONL file and
    formats them into a 'messages' structure.
    """
    valid_pairs_count = 0

    with open(input_filepath, "r", encoding="utf-8") as infile, open(
        output_filepath, "w", encoding="utf-8"
    ) as outfile:

        for line_number, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                article_data = json.loads(line)

                for pair in article_data.get("pairs", []):
                    question = pair.get("question")
                    answer = pair.get("answer")

                    if question and answer:
                        formatted_example = {
                            "messages": [
                                {"role": "user", "content": question.strip()},
                                {"role": "assistant", "content": answer.strip()},
                            ]
                        }

                        json.dump(formatted_example, outfile, ensure_ascii=False)
                        outfile.write("\n")
                        valid_pairs_count += 1

            except json.JSONDecodeError as e:
                print(f"Skipping malformed JSON on line {line_number}: {e}")

    print(
        f"Extraction complete. Successfully wrote {valid_pairs_count} formatted Q&A pairs to {output_filepath}."
    )


input_file = "pravda_pairs.jsonl"
output_file = "stance_tuning_dataset.jsonl"


if __name__ == "__main__":
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found")
    else:
        prepare_finetuning_data(input_file, output_file)
