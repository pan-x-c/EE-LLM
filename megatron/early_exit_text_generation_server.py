import datetime
import time
import torch
import json
import threading
import asyncio
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from megatron.text_generation import generate_and_post_process


GENERATE_NUM = 0
BEAM_NUM = 1
lock = threading.Lock()

class MegatronGenerate(Resource):
    def __init__(self, model):
        self.model = model
        asyncio.set_event_loop(asyncio.new_event_loop())
        self.loop = asyncio.get_event_loop()

    @staticmethod
    def send_do_generate():
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)
     
    @staticmethod
    def send_do_beam_search():
        choice = torch.cuda.LongTensor([BEAM_NUM])
        torch.distributed.broadcast(choice, 0)

    def check(self, raw_req):
        if not 'prompts' in raw_req:
            return 'prompts argument required', 400
        if len(raw_req['prompts']) == 0:
            return "prompts is empty", 400
        if len(raw_req['prompts']) > 128:
            return "Maximum number of prompts is 128", 400

    async def generate(self, req):
        MegatronGenerate.send_do_generate()  # Tell other ranks we're doing generate
        start_time = time.time()
        response, response_seg, response_logprobs, _ = \
            generate_and_post_process(
            self.model,
            prompts=req['prompts'],
            tokens_to_generate=req['tokens_to_generate'],
            echo_prompts=req['echo_prompts'],
            return_output_log_probs=req['logprobs'],
            top_k_sampling=req['top_k'],
            top_p_sampling=req['top_p'],
            top_p_decay=req['top_p_decay'],
            top_p_bound=req['top_p_bound'],
            temperature=req['temperature'],
            add_BOS=req['add_BOS'],
            use_stop_tokens_for_early_termination=True,
            stop_token_ids=req['stop_sequences'],
            prevent_newline_after_colon=req['prevent_newline_after_colon'],
            random_seed=req['random_seed'],
            early_exit_thres=req['early_exit_thres'],
            use_early_exit=req['use_early_exit'],
            print_max_prob=req['print_max_prob'],
            exit_layers=req['exit_layers'])
        end_time = time.time()
        print(f"Response(use {end_time - start_time}s): " + str(response))
        return {
            "text": response,
            "segments": response_seg,
            "logprobs": response_logprobs,
            "requst_time": end_time - start_time
            }

    def put(self):
        raw_req = request.get_json()

        if not "prompts" in raw_req:
            return "prompts argument required", 400
        
        if "max_len" in raw_req:
            return "max_len is no longer used.  Replace with tokens_to_generate", 400
        
        if "sentences" in raw_req:
            return "sentences is no longer used.  Replace with prompts", 400

        if isinstance(raw_req["prompts"], str):
            raw_req['prompts'] = [raw_req['prompts']]

        if not isinstance(raw_req["prompts"], list):
            return "prompts is not a list of strings", 400

        if len(raw_req['prompts']) == 0:
            return "prompts is empty", 400
        
        if len(raw_req['prompts']) > 128:
            return "Maximum number of prompts is 128", 400
        
        if 'tokens_to_generate' in raw_req:
            if not isinstance(raw_req['tokens_to_generate'], int):
                return "tokens_to_generate must be an integer greater than 0"
            if raw_req['tokens_to_generate'] < 0:
                return "tokens_to_generate must be an integer greater than or equal to 0"
        else:
            raw_req['tokens_to_generate'] = 64

        logprobs = False
        if "logprobs" in raw_req:
            logprobs = raw_req["logprobs"]
            if not isinstance(logprobs, bool):
                return "logprobs must be a boolean value"
        else:
            raw_req['logprobs'] = False

        if raw_req['tokens_to_generate'] == 0 and not raw_req['logprobs']:
            print("tokens_to_generate=0 implies logprobs should be True")
            raw_req['logprobs'] = True
        
        if "echo_prompts" in raw_req:
            if not isinstance(raw_req['echo_prompts'], bool):
                return "echo_prompts must be a bool"
        else:
            raw_req['echo_prompts'] = False

        if "early_exit_thres" in raw_req:
            if not type(raw_req['early_exit_thres']) == float or type(raw_req['early_exit_thres']) == int:
                return 'early_exit_thres must be a postive float number'
        else:
            raw_req['early_exit_thres'] = 40.0

        if "print_max_prob" in raw_req:
            raw_req['print_max_prob'] = True
        else:
            raw_req['print_max_prob'] = False

        if "exit_layers" in raw_req:
            if not type(raw_req['exit_layers']) == list:
                return "exit_layers must be a list of int"
            else:
                for i in raw_req['exit_layers']:
                    if not type(i) == int:
                        return "exit_layers must be a list of int"
        else:
            raw_req['exit_layers'] = []

        top_k = 0.0
        if "top_k" in raw_req:
            top_k = raw_req["top_k"]
            if not (type(top_k) == int):
                return "top_k must be an integer equal to or greater than 0 and less than or equal to 1000"
            if not (0 <= top_k <= 1000):
                return "top_k must be equal to or greater than 0 and less than or equal to 1000"
        else:
            raw_req['top_k'] = 0.0
        
        if "top_p" in raw_req:
            top_p = raw_req["top_p"]
            if not (type(top_p) == float or type(top_p) == int):
                return "top_p must be a positive float less than or equal to 1.0"
            if top_p > 0.0 and top_k > 0.0:
                return "cannot set both top-k and top-p samplings."
            if not (0 <= top_p <= 1.0):
                return "top_p must be less than or equal to 1.0"
        else:
            raw_req['top_p'] = 0.0
        
        if "top_p_decay" in raw_req:
            top_p_decay = raw_req["top_p_decay"]
            if not (type(top_p_decay) == float):
                return "top_p_decay must be a positive float less than or equal to 1.0"
            if top_p == 0.0:
                return "top_p_decay cannot be set without top_p"
            if not (0 <= top_p_decay <= 1.0):
                return "top_p_decay must be less than or equal to 1.0"
        else:
            raw_req['top_p_decay'] = 0.0

        top_p_bound = 0.0
        if "top_p_bound" in raw_req:
            top_p_bound = raw_req["top_p_bound"]
            if not (type(top_p_bound) == float):
                return "top_p_bound must be a positive float less than or equal to top_p"
            if top_p == 0.0:
                return "top_p_bound cannot be set without top_p"
            if not (0.0 < top_p_bound <= top_p):
                return "top_p_bound must be greater than 0 and less than top_p"
        else:
            raw_req['top_p_bound'] = 0.0

        if "temperature" in raw_req:
            temperature = raw_req["temperature"]
            if not (type(temperature) == int or type(temperature) == float):
                return "temperature must be a positive number less than or equal to 100.0"
            if not (0.0 <= temperature <= 100.0):
                return "temperature must be a positive number less than or equal to 100.0"
        else:
            raw_req['temperature'] = 0.0

        if raw_req['temperature'] == 0.0:
            raw_req['top_k'] = 1
            raw_req['top_p'] = 0

        if "add_BOS" in raw_req:
            if not isinstance(raw_req["add_BOS"], bool):
                return "add_BOS must be a boolean value"
        else:
            raw_req['add_BOS'] = False
        
        if any([len(prompt) == 0 for prompt in raw_req['prompts']]) and not raw_req["add_BOS"]:
            return "Empty prompts require add_BOS=true"

        if "stop_sequences" in raw_req:
            if not isinstance(raw_req["stop_sequences"], list):
                return "stop_sequences must be a str list"
            for seq in raw_req['stop_sequences']:
                if not isinstance(seq, str):
                    return "stop_sequences must be a str list"
        else:
            raw_req["stop_sequences"] = None

        if "prevent_newline_after_colon" in raw_req:
            if not isinstance(raw_req["prevent_newline_after_colon"], bool):
                return "prevent_newline_after_colon must be a boolean value"
        else:
            raw_req['prevent_newline_after_colon'] = False

        if "random_seed" in raw_req:
            random_seed = raw_req["random_seed"]
            if not isinstance(random_seed, int):
                return "random_seed must be integer"
            if random_seed < 0: 
                return "random_seed must be a positive integer"
        else:
            raw_req['random_seed'] = 1234

        if "use_early_exit" in raw_req:
            raw_req['use_early_exit'] = True
        else:
            raw_req['use_early_exit'] = False

        no_log = False
        if "no_log" in raw_req:
            no_log = raw_req["no_log"]
            if not isinstance(no_log, bool):
                return "no_log must be a boolean value"
        
        beam_width = None
        if "beam_width" in raw_req:
            beam_width = raw_req["beam_width"]
            if not isinstance(beam_width, int):
                return "beam_width must be integer"
            if beam_width < 1:
                return "beam_width must be an integer > 1"
            if len(raw_req['prompts']) > 1:
                return "When doing beam_search, batch size must be 1"

        if "length_penalty" in raw_req:
            length_penalty = raw_req["length_penalty"]
            if not isinstance(length_penalty, float):
                return "length_penalty must be a float"
        else:
            raw_req['length_penalty'] = 1

        if not no_log:
            print("request IP: " + str(request.remote_addr))
            print(json.dumps(raw_req),flush=True)
            print("start time: ", datetime.datetime.now())
        try:
            result = self.loop.run_until_complete(self.generate(raw_req))
            return jsonify(result)
        except ValueError as ve:
            return ve.args[0]


class MegatronServer(object):
    def __init__(self, model):
        self.app = Flask(__name__, static_url_path='')
        api = Api(self.app)
        api.add_resource(MegatronGenerate, '/api', resource_class_args=[model])
        
    def run(self, host, port):
        self.app.run(host=host, port=port, threaded=True, debug=False)
