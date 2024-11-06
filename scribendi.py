import os.path
from transformers import AutoTokenizer, AutoModelForCausalLM
from fuzzywuzzy.fuzz import token_sort_ratio
import torch
import argparse
from typing import List, Dict, Tuple, Optional
import csv

class ScribendiScore:
    def __init__(self, 
        threshold: float=0.8,
        model_id: str='gpt2',
        no_cuda: bool=False,
        access_token: Optional[str]=None,
    ) -> None:
        self.threshold = threshold
        self.model_id = model_id
        self.no_cuda = no_cuda
        self.tokenizer, self.model = self.load_model(model_id, access_token)

    def score(self,
              src_essays: Dict[str, List[str]],
              pred_essays: Dict[str, List[str]],
              batch_size: int = 4,
              verbose: bool = False
              ) -> int:
        score = 0
        score2freq = {-1: 0, 0: 0, 1: 0}  # Frequency tracker for scores

        # Collect essays into batches
        essay_ids = list(src_essays.keys())
        for batch_start in range(0, len(essay_ids), batch_size):
            batch_ids = essay_ids[batch_start:batch_start + batch_size]

            # Collect sentences for the current batch
            src_sents = [src_essays[essay_id][0] for essay_id in batch_ids if essay_id in src_essays]
            pred_sents = [pred_essays[essay_id][0] for essay_id in batch_ids if essay_id in pred_essays]

            # Compute perplexities for both source and prediction batches
            src_ppls = self.ppl(src_sents, batch_size)
            pred_ppls = self.ppl(pred_sents, batch_size)

            # Score each essay in the batch
            for i, essay_id in enumerate(batch_ids):
                src = src_sents[i]
                pred = pred_sents[i]

                # Skip if identical
                if src == pred:
                    score2freq[0] += 1
                    continue

                # Score essay based on perplexity and similarity metrics
                if src_ppls[i] <= pred_ppls[i]:
                    essay_score = -1
                    score2freq[-1] += 1
                else:
                    tsr = self.token_sort_ratio(src, pred)
                    ldr = self.levenshtein_distance_ratio(src, pred)
                    if max(tsr, ldr) >= self.threshold:
                        essay_score = 1
                        score2freq[1] += 1
                    else:
                        essay_score = -1
                        score2freq[-1] += 1

                score += essay_score
                if verbose:
                    print(f"Essay ID: {essay_id} -> Essay score: {essay_score}")

        # Print overall score frequency and normalized score
        if verbose:
            print('Overall score2freq ->', score2freq)
            print('Overall score ->', score2freq[1] - score2freq[-1])
        return score
                
    def ppl(self, sents: List[str], batch_size: int=4) -> List[int]:
        ppls = []
        sents = [self.tokenizer.bos_token + sent for sent in sents]
        for i in range(len(sents)//batch_size+1):
            batch = sents[i*batch_size:(i+1)*batch_size]
            if len(batch) == 0:
                continue
            inputs = self.tokenizer(batch, return_tensors='pt', padding=True)
            if not self.no_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    labels=inputs['input_ids']
                )
                shift_logits = outputs.logits[:, :-1, :].contiguous()
                shift_labels = inputs['input_ids'][:, 1:].contiguous()
                shift_mask = inputs['attention_mask'][:, 1:].contiguous()
                batch_size, seq_len = shift_labels.shape
                loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
                loss = loss_fn(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                ).view(batch_size, seq_len)
                loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
                ppls += torch.exp(loss).tolist()
        return ppls

    @staticmethod
    def token_sort_ratio(src: str, pred: str) -> float:
        return token_sort_ratio(src, pred) / 100

    @staticmethod
    def levenshtein_distance_ratio(src: str, pred: str) -> float:
        len_src = len(src)
        len_pred = len(pred)
        dp = [[0]*(len_pred+1) for _ in range(len_src+1)]
        # dp = np.zeros((len_src+1, len_pred+1))
        for i in range(1, len_src+1):
            dp[i][0] = i
        for j in range(1, len_pred+1):
            dp[0][j] = j
        for i in range(1, len_src+1):
            for j in range(1, len_pred+1):
                cost = 0
                if src[i-1] != pred[j-1]:
                    cost = 2 # Replacement cost is 2
                dp[i][j] = min(
                    dp[i-1][j-1] + cost,
                    min(dp[i-1][j] + 1, dp[i][j-1] + 1)
                )
        return 1 - dp[len_src][len_pred] / (len_src + len_pred)

    def load_model(
            self,
            model_id: str,
            access_token: Optional[str]
    ):
        local=os.path.exists(model_id)
        tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=local,
                token=access_token)
        model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype = torch.bfloat16,
                local_files_only=local,
                token=access_token)
        tokenizer.pad_token = tokenizer.eos_token
        if not self.no_cuda:
            model.to('cuda')
        return tokenizer, model

    @staticmethod
    def remove_eq_sents(
        src_sents: List[str],
        pred_sents: List[str]
    )-> Tuple[List[str], List[str], int]:
        new_src_sents = []
        new_pred_sents = []
        count = 0
        for src, pred in zip(src_sents, pred_sents):
            if src != pred:
                new_src_sents.append(src)
                new_pred_sents.append(pred)
            else:
                count += 1
        return new_src_sents, new_pred_sents, count

def load_file(file_path: str) -> List[str]:
    sentences = []
    with open(file_path) as fp:
        for line in fp:
            sent = line.rstrip()
            sentences.append(sent)
    return sentences

def load_multi_gec_file(file_path: str) -> Dict[str, List[str]]:
    essays = {}
    current_essay_id = None
    if file_path.endswith(".md"):
        with open(file_path) as fp:
            for line in fp:
                line = line.strip()
                if not line: continue
                if line.startswith("### essay_id = "):
                    current_essay_id = line.split(" = ")[1]
                    essays[current_essay_id] = []
                elif current_essay_id:
                    essays[current_essay_id].append(line)
    elif file_path.endswith(".tmp"):
        with open(file_path) as fp:
            for line_id, line in enumerate(fp):
                line = line.strip()
                essays[line_id] = [line]
    else:
        raise Exception("wrong file format!")
    return essays

def extract_lang_corpus(src_file):
    """
    Extract language and corpus from the src_file name.
    Assumes src_file names follow the format:
    <lang>-<corpus>-orig-dev.tmp
    """
    basename = os.path.basename(src_file)
    parts = basename.split('-')
    if len(parts) >= 2:
        language = parts[0]
        corpus = parts[1]
        return language, corpus
    return None, None


def process_directory_pairs(ref_dir, res_dir, team_name, scorer, batch_size, verbose,split="dev"):
    with open('scorer_scribendi.csv', 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        if os.stat('scorer_scribendi.csv').st_size == 0:
            csv_writer.writerow(['team_name', 'language', 'corpus', 'normalized_score'])

        for ref_file in os.listdir(ref_dir):
            if "orig" in ref_file and split in ref_file:
                base_name = ref_file.replace("-orig-dev.tmp", "")
                pred_file = os.path.join(res_dir, f"{base_name}-hypo1-{split}.tmp")
                if not os.path.exists(pred_file):
                    pred_file= pred_file.replace("hypo1", "hypo")
                    assert os.path.exists(pred_file)
                ref_path = os.path.join(ref_dir, ref_file)
                if os.path.isfile(pred_file):
                    print(f"Running scorer on pair: {ref_path} and {pred_file}")
                    src_essays = load_multi_gec_file(ref_path)
                    pred_essays = load_multi_gec_file(pred_file)

                    language, corpus = extract_lang_corpus(ref_path)
                    score = scorer.score(src_essays, pred_essays, batch_size=batch_size, verbose=verbose)
                    normalized_score = score / len(pred_essays) if len(pred_essays) > 0 else 0
                    print(f'Absolute: {score}  Normalized: {score / len(pred_essays):.4g}')
                    csv_writer.writerow([team_name, language, corpus, f'{normalized_score:.4g}'])
                else:
                    print(f"Warning: Prediction file {pred_file} does not exist for {ref_file}")


def main(args):
    scorer = ScribendiScore(
        model_id=args.model_id,
        threshold=args.threshold,
        no_cuda=args.no_cuda,
        access_token=args.access_token
    )

    if os.path.isdir(args.src) and os.path.isdir(args.pred):
        process_directory_pairs(args.src, args.pred, args.team_name, scorer, args.batch_size, args.verbose)
    elif os.path.isfile(args.src) and os.path.isfile(args.pred):
        src_file = args.src
        pred_file = args.pred
        src_essays = load_multi_gec_file(src_file)
        pred_essays = load_multi_gec_file(pred_file)
        print(src_file, pred_file)
        score = scorer.score(src_essays, pred_essays, batch_size=args.batch_size, verbose=args.verbose)
        print(src_file, pred_file)
        print(f'Absolute: {score}  Normalized: {score/len(pred_essays):.4g}')


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src')
    parser.add_argument('--pred')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--model_id', default='meta-llama/Llama-3.1-8B')
    parser.add_argument('--access_token', default=None)
    parser.add_argument('--team_name', default='team1')
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_parser()
    main(args)
