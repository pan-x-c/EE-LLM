import requests
import json
import time

URL = "http://localhost:5000/api"
HEADER = {
    "Content-Type": "application/json; charset=UTF-8",
}


def request(
    prompts,
    tokens_to_generate=100,
    use_early_exit=True,
    early_exit_thres=0.8,
    print_max_prob=False,
    exit_layers=[]
):
    length = len(prompts)
    for i in range(length):
        data = {
            "prompts": [prompts[i]],
            "tokens_to_generate": tokens_to_generate,
            "top_k": 1,
            "logprobs": True,
            "random_seed": int(time.time_ns()) % 16384,
            "echo_prompts": False,
            "early_exit_thres": early_exit_thres,
            "exit_layers": exit_layers
        }
        if use_early_exit:
            data["use_early_exit"] = True
        if print_max_prob:
            data["print_max_prob"] = True
        start_time = time.time()
        response = requests.put(URL, headers=HEADER, data=json.dumps(data))
        end_time = time.time()
        print("Request:-------------------------------------------------")
        print(f"{prompts[i]}")
        print(
            f"Response:------------------({end_time - start_time:.4f}s)-------------------"
        )
        try:
            print(f'{response.json()["text"][0]}')
        except Exception as e:
            print(response)
        print("----------------------------------------------------------")


def main(
    file_name, tokens_to_generate, use_early_exit, early_exit_thres, print_max_prob, exit_layers
):
    prompts = []
    with open(file_name, "r") as f:
        for line in f.readlines():
            prompts.append(json.loads(line)["text"])
    request(
        prompts, tokens_to_generate, use_early_exit, early_exit_thres, print_max_prob, exit_layers
    )


if __name__ == "__main__":
    main(
        "tools/prompt_example.jsonl",
        tokens_to_generate=50,
        use_early_exit=True,
        early_exit_thres=1.0,
        print_max_prob=True,
        exit_layers=[]
    )
