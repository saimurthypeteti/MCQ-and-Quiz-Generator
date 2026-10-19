"""Clean educational QA records and split by passage group without leakage."""
import argparse, csv, hashlib, json, random, re
from pathlib import Path

REQUIRED={"passage_id","passage","question","options","correct_answer"}

def normalize(s): return re.sub(r"\s+"," ",str(s or "")).strip()

def adapt(row,index):
    """Map common SciQ-style fields to the project's canonical MCQ schema."""
    if REQUIRED.issubset(row):return row
    passage=normalize(row.get("passage") or row.get("support") or row.get("context") or row.get("article"))
    question=normalize(row.get("question"))
    answer=normalize(row.get("correct_answer") or row.get("answer"))
    distractors=[normalize(row.get(k)) for k in ("distractor1","distractor2","distractor3") if normalize(row.get(k))]
    options=row.get("options") or ([answer]+distractors if answer else None)
    if isinstance(options,str):
        try:options=json.loads(options)
        except json.JSONDecodeError:options=[x.strip() for x in options.split("|") if x.strip()]
    if not passage:return row
    passage_id=normalize(row.get("passage_id") or row.get("id")) or hashlib.sha256(passage.encode()).hexdigest()[:20]
    return {"passage_id":passage_id,"passage":passage,"question":question,"options":options,"correct_answer":answer,"explanation":row.get("explanation"),"difficulty":row.get("difficulty"),"source_id":row.get("source_id") or row.get("id"),"source_page":row.get("source_page"),"source_section":row.get("source_section"),"annotation_source":row.get("annotation_source") or "human-written"}
def load(path):
    path=Path(path)
    if path.suffix.lower()==".csv":
        with path.open(encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    else:
        raw=path.read_text(encoding="utf-8-sig")
        try:
            obj=json.loads(raw); rows=obj if isinstance(obj,list) else [obj]
        except json.JSONDecodeError:
            rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    return rows

def clean(row):
    item={k:row.get(k) for k in REQUIRED|{"explanation","difficulty","source_id","source_page","source_section"}}
    for key in ("passage_id","passage","question","correct_answer"): item[key]=normalize(item.get(key))
    options=item.get("options")
    if isinstance(options,str):
        try: options=json.loads(options)
        except json.JSONDecodeError: options=[x.strip() for x in options.split("|") if x.strip()]
    if not isinstance(options,list): return None
    item["options"]=[normalize(x) for x in options if normalize(x)]
    for key in ("explanation","difficulty","source_id","source_page","source_section"):
        if item.get(key) is not None:item[key]=normalize(item[key])
    if not all(item.get(k) for k in ("passage_id","passage","question","correct_answer")):return None
    if len(set(x.casefold() for x in item["options"]))<2 or item["correct_answer"].casefold() not in {x.casefold() for x in item["options"]}:return None
    return item

def prepare(rows,seed=16):
    seen=set(); records=[]; invalid=duplicates=0
    for index,raw in enumerate(rows):
        row=adapt(raw,index)
        item=clean(row)
        if not item:invalid+=1;continue
        key=(item["passage_id"].casefold(),item["question"].casefold())
        if key in seen:duplicates+=1;continue
        seen.add(key);records.append(item)
    groups={}
    for rec in records:groups.setdefault(rec["passage_id"],[]).append(rec)
    ids=sorted(groups); random.Random(seed).shuffle(ids)
    n=len(ids); ntest=round(n*.15) if n>=4 else (1 if n else 0); nval=round(n*.15) if n>=5 else (1 if n>=3 else 0)
    if ntest+nval>=n and n>=3: ntest=nval=1
    split_ids={"test":set(ids[:ntest]),"validation":set(ids[ntest:ntest+nval]),"train":set(ids[ntest+nval:])}
    return {s:[r for r in records if r["passage_id"] in group] for s,group in split_ids.items()}, {"input_records":len(rows),"valid_records":len(records),"invalid_records":invalid,"duplicate_records":duplicates,"passage_groups":n,"split_passages":{s:len(g) for s,g in split_ids.items()},"split_records":{s:sum(len(groups[i]) for i in g) for s,g in split_ids.items()},"seed":seed,"group_key":"passage_id"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--output-dir",required=True);p.add_argument("--seed",type=int,default=16);p.add_argument("--dataset-name",default="");p.add_argument("--license",default="");p.add_argument("--source-url",default="");a=p.parse_args()
    rows=load(a.input);splits,manifest=prepare(rows,a.seed);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    for name,records in splits.items():
        with (out/f"{name}.jsonl").open("w",encoding="utf-8") as f:
            for row in records:f.write(json.dumps(row,ensure_ascii=False)+"\n")
    manifest.update(source_file=Path(a.input).name,dataset_name=a.dataset_name or Path(a.input).stem,license=a.license or "Unknown - verify before use",source_url=a.source_url or None)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))
if __name__=="__main__":main()
