"""Compare rule and local model MCQ generation on the same held-out records."""
import argparse,csv,json,re,time
from collections import Counter
from pathlib import Path

STOP=set("the a an and or but is are was were be been being to of in on at by for with from as into that this these those it its their they them he she we you can may will should must has have had not than then which who what when where how also such".split())

def words(text):return [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z'-]*",text or "") if w.lower() not in STOP]
def token_f1(pred,ref):
    a,b=words(pred),words(ref)
    if not a or not b:return 0.0
    common=sum((Counter(a)&Counter(b)).values())
    return 2*common/(len(a)+len(b)) if common else 0.0

def rule_generate(passage,choice_count=4):
    for sentence in re.split(r"(?<=[.!?])\s+|\n+",passage):
        tokens=re.findall(r"[A-Za-z][A-Za-z'-]*",sentence)
        candidates=[w for w in tokens if w.lower() not in STOP and len(w)>3]
        if not candidates:continue
        answer=max(dict.fromkeys(candidates),key=lambda w:(len(w),w[0].isupper()))
        question=f"Which term completes this statement? {re.sub(r'\\b'+re.escape(answer)+r'\\b','____',sentence,count=1,flags=re.I)}"
        distractors=[]
        for word in candidates:
            if word.lower()!=answer.lower() and word.lower() not in {x.lower() for x in distractors}:distractors.append(word)
        distractors += [x for x in ["concept","process","system","method","example"] if x.lower()!=answer.lower() and x.lower() not in {d.lower() for d in distractors}]
        return {"question":question,"options":[answer]+distractors[:max(1,choice_count-1)],"answer":answer,"explanation":f"The source states: {sentence}","source":sentence}
    return {"question":"","options":[],"answer":"","explanation":"","source":""}

def valid(item):
    options=item.get("options") or []; answer=str(item.get("answer") or "").strip()
    return bool(str(item.get("question") or "").strip() and len(options)>=2 and len({str(x).casefold() for x in options})==len(options) and answer.casefold() in {str(x).casefold() for x in options})

def supported(item,passage):return bool(item.get("answer") and str(item["answer"]).casefold() in passage.casefold())
def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

def load_model(name,local_only):
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,pipeline
    tok=AutoTokenizer.from_pretrained(name,local_files_only=local_only)
    model=AutoModelForSeq2SeqLM.from_pretrained(name,local_files_only=local_only)
    return pipeline("text2text-generation",model=model,tokenizer=tok,device=-1)

def model_generate(pipe,passage,choice_count=4):
    prompt=("Write one educational single-answer multiple-choice question answerable from the passage. "
      f"Return only JSON with keys question, options (exactly {choice_count} strings), answer (one option), explanation. "
      "Do not add facts absent from the passage. Passage: "+passage[:4000])
    raw=pipe(prompt,max_new_tokens=240,do_sample=False)[0]["generated_text"]
    match=re.search(r"\{.*\}",raw,re.S)
    if not match:raise ValueError("Model output did not contain a JSON object")
    result=json.loads(match.group()); result["source"]=passage
    return result

def load_similarity_model(name,local_only):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name,local_files_only=local_only)

def metric_summary(outputs,records,embeddings=None):
    n=len(records)
    validity=sum(valid(o) for o in outputs)/n if n else None
    answerability=sum(supported(o,r["passage"]) for o,r in zip(outputs,records))/n if n else None
    token_overlap=sum(token_f1(o.get("question",""),r["question"]) for o,r in zip(outputs,records))/n if n else None
    sem=(sum(float(embeddings[i][0]@embeddings[i][1]) for i in range(n))/n) if embeddings and n else None
    dup=0
    if embeddings and n>1:
        dup=sum(any(float(embeddings[i][0]@embeddings[j][0])>=.9 for j in range(i)) for i in range(1,n))/n
    return {"validity_rate":round(validity,4) if validity is not None else None,"answerability_rate":round(answerability,4) if answerability is not None else None,"verbatim_answer_support_rate":round(answerability,4) if answerability is not None else None,"semantic_similarity_to_human_question":round(sem,4) if sem is not None else None,"semantic_duplicate_rate_at_cosine_0_90":round(dup,4) if embeddings and n else None,"token_f1_to_human_question":round(token_overlap,4) if token_overlap is not None else None}

def main():
    p=argparse.ArgumentParser();p.add_argument("--test",required=True);p.add_argument("--output",required=True);p.add_argument("--model",default="");p.add_argument("--similarity-model",default="sentence-transformers/all-MiniLM-L6-v2");p.add_argument("--no-semantic",action="store_true");p.add_argument("--local-files-only",action=argparse.BooleanOptionalAction,default=True);a=p.parse_args()
    records=read_jsonl(a.test);outdir=Path(a.output);outdir.mkdir(parents=True,exist_ok=True)
    if not records:raise SystemExit("Held-out test data is empty; no evaluation was run.")
    model_pipe=None;model_info={"name":a.model or None,"status":"not requested" if not a.model else "unavailable"}
    if a.model:
        try:model_pipe=load_model(a.model,a.local_files_only);model_info["status"]="loaded"
        except Exception as exc:model_info["error"]=f"{type(exc).__name__}: {exc}"
    semantic_model=None;semantic_info={"name":a.similarity_model,"status":"disabled" if a.no_semantic else "unavailable"}
    if not a.no_semantic:
        try:semantic_model=load_similarity_model(a.similarity_model,a.local_files_only);semantic_info["status"]="loaded"
        except Exception as exc:semantic_info["error"]=f"{type(exc).__name__}: {exc}"
    rows=[];rule_outputs=[];model_outputs=[];latencies=[];rule_latencies=[];model_errors=[]
    for rec in records:
        rule_start=time.perf_counter();rule=rule_generate(rec["passage"],len(rec.get("options",[])) or 4);rule_latency=(time.perf_counter()-rule_start)*1000;rule_latencies.append(rule_latency);rule_outputs.append(rule)
        model_out=None;latency=None
        if model_pipe:
            start=time.perf_counter()
            try:model_out=model_generate(model_pipe,rec["passage"],len(rec.get("options",[])) or 4);latency=(time.perf_counter()-start)*1000;model_outputs.append(model_out);latencies.append(latency)
            except Exception as exc:model_errors.append(f"{type(exc).__name__}: {exc}");model_outputs.append({"question":"","options":[],"answer":"","source":rec["passage"]})
        rows.append({"passage_id":rec["passage_id"],"reference_question":rec["question"],"rule_question":rule["question"],"rule_options":json.dumps(rule["options"],ensure_ascii=False),"rule_answer":rule["answer"],"rule_valid":valid(rule),"rule_answerable":supported(rule,rec["passage"]),"rule_token_f1":round(token_f1(rule["question"],rec["question"]),4),"rule_latency_ms":round(rule_latency,2),"model_question":model_out.get("question","") if model_out else "","model_options":json.dumps(model_out.get("options",[]),ensure_ascii=False) if model_out else "","model_answer":model_out.get("answer","") if model_out else "","model_valid":valid(model_out) if model_out else None,"model_answerable":supported(model_out,rec["passage"]) if model_out else None,"model_token_f1":round(token_f1(model_out.get("question",""),rec["question"]),4) if model_out else None,"model_latency_ms":round(latency,2) if latency is not None else None})
    semantic_scores={}
    if semantic_model:
        for method,outputs in (("rule",rule_outputs),("model",model_outputs if model_pipe else [])):
            if not outputs:continue
            texts=[x.get("question","") for x in outputs]
            try:
                vectors=semantic_model.encode([t for pair in zip(texts,[r["question"] for r in records]) for t in pair],normalize_embeddings=True,convert_to_numpy=True)
                pairs=[(vectors[2*i],vectors[2*i+1]) for i in range(len(records))]
                semantic_scores[method]=pairs
            except Exception as exc:semantic_info["error"]=f"{type(exc).__name__}: {exc}";semantic_info["status"]="failed"
    manifest_path=Path(a.test).parent/"manifest.json";manifest=json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    summary={"test_records":len(records),"test_file":Path(a.test).name,"same_heldout_examples_for_methods":True,"dataset_name":manifest.get("dataset_name",records[0].get("dataset_name")),"dataset_license":manifest.get("license",records[0].get("license","Record license/provenance in the dataset manifest")),"dataset_source":manifest.get("source_url"),"rule_baseline":metric_summary(rule_outputs,records,semantic_scores.get("rule")),"rule_generation_latency_ms_mean":round(sum(rule_latencies)/len(rule_latencies),2) if rule_latencies else None,"model_based":metric_summary(model_outputs,records,semantic_scores.get("model")) if model_pipe else None,"model":model_info,"model_item_errors":model_errors,"semantic_model":semantic_info,"model_generation_latency_ms_mean":round(sum(latencies)/len(latencies),2) if latencies else None,"metric_notes":{"validity":"Structural MCQ check: question, at least two distinct options, and answer among options.","answerability":"Estimated by verbatim occurrence of the generated answer in the source passage; not entailment.","semantic_similarity":"Cosine similarity of generated and human-written question embeddings; only populated when Sentence Transformers model actually loads.","token_f1":"Lexical overlap diagnostic; not factual correctness.","duplicate_rate":"Within-method generated-question cosine similarity threshold 0.90."},"human_annotation":"Dataset annotations are preserved; no extra student evaluation labels are invented."}
    with (outdir/"predictions.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=list(rows[0]);writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (outdir/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8");print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
