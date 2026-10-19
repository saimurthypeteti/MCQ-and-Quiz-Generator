"""Streamlit pages for the local Quill Studio API."""
from __future__ import annotations
import json, os, threading, urllib.error, urllib.request
from pathlib import Path
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]


def api_url():
    return os.getenv("MCQ_API_URL",f"http://{os.getenv('HOST','127.0.0.1')}:{os.getenv('PORT','8000')}/api")


@st.cache_resource
def ensure_backend():
    """Start the existing local API in this process if it is not already running."""
    try:
        request("GET","/health")
        request("GET","/rag/status")
        return "connected"
    except Exception:
        import sys
        sys.path.insert(0,str(ROOT))
        from backend import app as service
        service.init_db()
        host=os.getenv("HOST","127.0.0.1");server=None
        preferred=int(os.getenv("PORT","8000"))
        for port in [preferred,*range(preferred+1,preferred+11)]:
            try:server=service.ThreadingHTTPServer((host,port),service.API);break
            except OSError:continue
        if server is None:raise RuntimeError("Could not bind the local quiz API on the configured port or the next ten ports.")
        os.environ["PORT"]=str(server.server_address[1])
        threading.Thread(target=server.serve_forever,daemon=True,name="quill-api").start()
        return server


def request(method,path,payload=None,content_type="application/json"):
    url=api_url()+path
    body=None if payload is None else (payload if isinstance(payload,bytes) else json.dumps(payload,ensure_ascii=False).encode())
    req=urllib.request.Request(url,data=body,headers={"Content-Type":content_type} if body is not None else {},method=method)
    try:
        with urllib.request.urlopen(req,timeout=120) as response:
            raw=response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        try: detail=json.loads(exc.read().decode()).get("error",str(exc))
        except Exception: detail=str(exc)
        raise RuntimeError(detail) from exc


def multipart_upload(name,data):
    boundary="----QuillStudioBoundary7MA4YWxkTrZu0gW"
    head=(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{Path(name).name}\"\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
    body=head+data+f"\r\n--{boundary}--\r\n".encode()
    return request("POST","/documents",body,f"multipart/form-data; boundary={boundary}")


def page_shell(title,eyebrow,description):
    ensure_backend()
    st.markdown("""<style>
    :root{--ink:#17213d;--muted:#74809a;--line:#e5e9f2;--brand:#5865e8;--brand-dark:#414cc5;--surface:#fff}
    .stApp{background:radial-gradient(ellipse at 80% 0%,rgba(222,226,255,.52),transparent 34%),#f6f7fb;color:var(--ink)}
    [data-testid="stHeader"]{background:rgba(246,247,251,.84);backdrop-filter:blur(12px)}
    [data-testid="stSidebar"]{background:linear-gradient(180deg,#fff 0%,#f8f9ff 100%);border-right:1px solid var(--line)}
    [data-testid="stSidebar"] [data-testid="stSidebarNav"]{padding-top:.7rem}
    .block-container{padding:2rem 2.15rem 4rem;max-width:1320px}
    .hero{position:relative;isolation:isolate;overflow:hidden;padding:1.8rem 2rem;border-radius:22px;background:linear-gradient(112deg,#17213d 0%,#303d96 58%,#5865e8 100%);box-shadow:0 18px 42px rgba(42,52,126,.18);margin:0 0 1.6rem;color:#fff}
    .hero:after{content:"";position:absolute;z-index:-1;width:280px;height:280px;right:7%;top:-185px;border:1px solid rgba(255,255,255,.2);border-radius:50%;box-shadow:0 0 0 30px rgba(255,255,255,.045),0 0 0 65px rgba(255,255,255,.035)}
    .eyebrow{display:inline-flex;align-items:center;gap:.45rem;font-size:.69rem;font-weight:800;letter-spacing:.15em;text-transform:uppercase;color:#c9d0ff;background:rgba(255,255,255,.1);border:1px solid rgba(255,255,255,.14);padding:.38rem .65rem;border-radius:999px}
    .hero h1{margin:.75rem 0 .38rem;font-size:clamp(1.8rem,3.3vw,2.5rem);line-height:1.12;letter-spacing:-.035em;color:#fff}
    .hero .muted{max-width:720px;color:rgba(239,242,255,.78);font-size:1rem;line-height:1.55}
    h2,h3{color:var(--ink);letter-spacing:-.02em} h2{margin-top:1.6rem}
    .stMetric{background:linear-gradient(145deg,#fff,#fafbff);border:1px solid var(--line);border-radius:16px;padding:1rem 1.05rem;box-shadow:0 5px 18px rgba(24,35,69,.045);transition:transform .18s ease,box-shadow .18s ease}
    .stMetric:hover{transform:translateY(-2px);box-shadow:0 10px 24px rgba(24,35,69,.09)}
    [data-testid="stMetricLabel"]{color:var(--muted);font-weight:600}[data-testid="stMetricValue"]{color:var(--ink);font-weight:750;letter-spacing:-.035em}
    .stButton>button,.stFormSubmitButton>button{min-height:2.7rem;border-radius:10px;border:1px solid #dfe3f0;background:#fff;color:#303955;font-weight:650;transition:all .16s ease}
    .stButton>button:hover,.stFormSubmitButton>button:hover{border-color:#aab2f7;color:var(--brand-dark);transform:translateY(-1px);box-shadow:0 7px 16px rgba(61,75,190,.12)}
    .stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{background:linear-gradient(110deg,#5865e8,#4653d0);border:1px solid #5865e8;color:#fff;box-shadow:0 6px 15px rgba(74,87,211,.2)}
    .stButton>button[kind="primary"]:hover,.stFormSubmitButton>button[kind="primary"]:hover{background:linear-gradient(110deg,#4a57d8,#3744bc);color:#fff;box-shadow:0 9px 20px rgba(74,87,211,.28)}
    [data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,[data-testid="stSelectbox"]>div>div{border-radius:10px!important}
    [data-testid="stTextInput"] input:focus,[data-testid="stTextArea"] textarea:focus{border-color:#7c87ef;box-shadow:0 0 0 3px rgba(88,101,232,.13)}
    [data-testid="stWidgetLabel"] p{font-size:.88rem;font-weight:650;color:#39445f}
    [data-testid="stExpander"]{border:1px solid var(--line);border-radius:13px;background:#fff;box-shadow:0 3px 12px rgba(24,35,69,.035);overflow:hidden}
    [data-testid="stExpander"] summary{font-weight:650;color:#27324f}
    [data-testid="stFileUploader"] section{border:1px dashed #aeb7e9;border-radius:14px;background:linear-gradient(145deg,#fff,#f7f8ff)}
    [data-testid="stAlert"]{border-radius:12px}
    [data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
    .source{border-left:3px solid #6976e5;background:#f2f4ff;padding:.8rem 1rem;border-radius:0 9px 9px 0;color:#454d67}
    @media(max-width:760px){.block-container{padding:1.15rem 1rem 3rem}.hero{padding:1.35rem 1.25rem;border-radius:17px}.hero:after{right:-100px}.stMetric{padding:.8rem}}
    @media(prefers-reduced-motion:reduce){*,*:before,*:after{transition:none!important;scroll-behavior:auto!important}}
    </style>""",unsafe_allow_html=True)
    st.markdown(f'<div class="hero"><div class="eyebrow">✦ &nbsp; {eyebrow}</div><h1>{title}</h1><div class="muted">{description}</div></div>',unsafe_allow_html=True)


def page_dashboard():
    page_shell("Your learning workspace","QUIll STUDIO · P16","Grounded quizzes from your course materials.")
    d=request("GET","/dashboard"); cols=st.columns(4)
    for col,label,key in zip(cols,["Documents processed","Quizzes saved","Questions generated","Quiz attempts"],["documents","quizzes","questions","attempts"]):col.metric(label,d[key])
    c1,c2=st.columns([1.3,1]);
    with c1:
        st.subheader("Recent quizzes")
        for quiz in d.get("recent",[]):st.write(f"**{quiz['title']}** · {quiz['question_count']} questions · {quiz['status'].title()}")
        if not d.get("recent"):st.info("No quizzes yet. Upload course material to begin.")
    with c2:
        st.subheader("Average score")
        st.metric("Across completed attempts",f"{d['average_score']:.1f}%" if d["attempts"] else "—")
        st.caption("Counts and scores come from the local SQLite database.")


def page_upload():
    page_shell("Upload course material","MATERIALS","Extract text, preview it, and build a document-specific retrieval index.")
    uploaded=st.file_uploader("Choose a PDF, DOCX, or TXT file",type=["pdf","docx","txt"],max_upload_size=15)
    pasted=st.text_area("Or paste study material",height=160,placeholder="Paste at least 40 characters…")
    if st.button("Process material",type="primary",disabled=not(uploaded or pasted.strip())):
        with st.spinner("Extracting text and building the chunk index…"):
            if uploaded:
                doc=multipart_upload(uploaded.name,uploaded.getvalue())
            else:
                doc=request("POST","/documents/text",{"name":"Pasted study material.txt","text":pasted})
            st.session_state.active_document=doc
        st.success(f"Saved {doc['name']} · {doc.get('index',{}).get('chunks',0)} indexed chunks · {doc.get('index',{}).get('status','unknown')}")
    docs=request("GET","/documents")
    st.subheader("Your library")
    if docs:
        for doc in docs:
            with st.expander(f"{doc['name']} · {len(doc['text']):,} characters"):
                st.text_area("Extracted text preview",doc["text"],height=220,key="preview_"+doc["id"])
    else:st.info("No materials uploaded yet.")


def page_generate():
    page_shell("Build a topic-focused quiz","ASSESSMENT BUILDER","Retrieve relevant source chunks first, then generate a reviewable quiz.")
    docs=request("GET","/documents")
    if not docs:st.warning("Upload a document first.");return
    labels={f"{d['name']} · {d['id'][:8]}":d for d in docs}
    with st.form("generation"):
        source_label=st.selectbox("Source document",list(labels)); title=st.text_input("Quiz title",Path(labels[source_label]["name"]).stem)
        topic=st.text_input("Topic, chapter, or learning objective",placeholder="e.g. how chlorophyll absorbs light")
        objectives=st.text_input("Additional learning objectives (optional)",placeholder="Explain energy conversion; distinguish light and dark reactions")
        cols=st.columns(4); count=cols[0].selectbox("Questions",[5,10,15,20]);difficulty=cols[1].selectbox("Difficulty",["Easy","Medium","Hard"]);qtype=cols[2].selectbox("Type",["mcq","truefalse"],format_func=lambda x:"Multiple choice" if x=="mcq" else "True / False"); choices=cols[3].selectbox("Options",[3,4,5,6],index=1)
        include_explanations=st.checkbox("Include explanations",value=True); academic_level=st.selectbox("Academic level",["General","High school","Undergraduate","Advanced"])
        submitted=st.form_submit_button("Retrieve chunks & generate",type="primary")
    if submitted:
        try:
            payload={"title":title,"document_id":labels[source_label]["id"],"topic":topic,"learning_objectives":objectives,"academic_level":academic_level,"count":count,"difficulty":difficulty,"type":qtype,"choices":choices,"explanations":include_explanations}
            with st.spinner("Searching the document and generating questions…"):quiz=request("POST","/generate",payload)
            st.session_state.review_quiz=quiz;st.success(f"Generated {len(quiz['questions'])} questions. Retrieval: {(quiz.get('retrieval') or {}).get('method','full passage')}.")
            used_generator=(quiz.get("settings") or {}).get("generator","rules")
            if used_generator=="model":
                fallback_count=sum(any("rule-based" in flag.casefold() for flag in q.get("quality",{}).get("flags",[])) for q in quiz["questions"])
                model_count=len(quiz["questions"])-fallback_count
                if fallback_count:st.warning(f"Question stems: {model_count} generated by FLAN-T5 and {fallback_count} supplied by the rules fallback. Each question is labeled below. Review the answers and choices before use.")
                else:st.success(f"Question stems: all {model_count} generated by FLAN-T5. Answers and choices are grounded or checked against the source; review each question before use.")
            else:st.info("Question generator: rules-based. To use the pretrained model, choose it in Settings and click Apply for this Streamlit session before generating.")
            st.subheader("Retrieved sources")
            for i,q in enumerate(quiz["questions"],1):
                with st.expander(f"Q{i}. {q['question']}"):
                    st.write("**Answer:**",q["answer"]);st.write("**Section:**",q.get("source_section") or "Source document");st.write("**Excerpt:**",q.get("source",""));st.caption(f"Retrieval score: {q.get('retrieval_score')}")
                    flags=q.get("quality",{}).get("flags",[])
                    if flags:st.caption("Quality notes: "+" · ".join(flags))
        except Exception as exc:st.error(str(exc))


def page_review():
    page_shell("Review and edit questions","QUESTION REVIEW","Check the answer choices and cited source before saving or publishing.")
    quizzes=request("GET","/quizzes")
    q=st.session_state.get("review_quiz")
    if not q and quizzes:
        choices={f"{x['title']} · {x['id'][:8]}":x for x in quizzes}; selected=st.selectbox("Open a saved quiz",list(choices));
        if st.button("Load quiz"):st.session_state.review_quiz=request("GET",f"/quizzes/{choices[selected]['id']}");st.rerun()
        return
    if not q:st.info("Generate or load a quiz to review it.");return
    for i,item in enumerate(q["questions"]):
        widget_id=f"{q['id']}_{i}"
        saved_options=item.get("options",[])
        if item.get("type")!="truefalse" and (len(saved_options)<2 or any(not str(option).strip() for option in saved_options)):
            st.warning(f"Question {i+1} has a missing answer choice. Click Regenerate question {i+1} to rebuild its choices from the source passage.")
        if st.button(f"↻ Regenerate question {i+1}",key=f"regen_{q['id']}_{i}"):
            try:
                replacement=request("POST","/questions/regenerate",{"source":item.get("source",""),"difficulty":item.get("difficulty","Medium"),"type":item.get("type","mcq"),"choices":len(item.get("options",[])) or 4,"topic":item.get("topic",""),"avoid_answer":item.get("answer","") if item.get("type")!="truefalse" else "","avoid_question":item.get("question","")})["question"]
                for suffix in ("rq_", "ra_", "re_", "rs_", "rk_"):
                    st.session_state.pop(f"{suffix}{widget_id}",None)
                for j in range(6):st.session_state.pop(f"ro_{widget_id}_{j}",None)
                q["questions"][i]=replacement;st.session_state.review_quiz=q;st.rerun()
            except Exception as exc:st.error(str(exc))
    with st.form("review_form"):
        title=st.text_input("Quiz title",q["title"]); updated=[]
        for i,item in enumerate(q["questions"]):
            # This form uses per-question widget keys, so derive the identity
            # inside the loop (the earlier regeneration loop has its own i).
            widget_id=f"{q['id']}_{i}"
            with st.expander(f"Question {i+1}",expanded=i==0):
                question=st.text_input("Question",item["question"],key=f"rq_{widget_id}")
                opts=item.get("options",[]); options=[]
                for j in range(max(2,len(opts))):options.append(st.text_input(f"Option {j+1}",opts[j] if j<len(opts) else "",key=f"ro_{widget_id}_{j}"))
                answer=st.selectbox("Correct answer",options,index=options.index(item["answer"]) if item.get("answer") in options else 0,key=f"ra_{widget_id}")
                explanation=st.text_area("Explanation",item.get("explanation",""),key=f"re_{widget_id}")
                st.caption(f"Source: {item.get('source_section') or 'Passage'} · {item.get('source','')}")
                status=st.selectbox("Review decision",["pending","approved","rejected"],index=["pending","approved","rejected"].index(item.get("review_status","pending")),key=f"rs_{widget_id}")
                keep=st.checkbox("Keep this question",value=True,key=f"rk_{widget_id}")
                updated.append({**item,"question":question,"options":[x for x in options if x.strip()],"answer":answer,"explanation":explanation,"review_status":status,"_keep":keep})
        action=st.radio("Save as",["Draft","Approve and publish"],horizontal=True)
        st.caption("Rejected questions are removed when you save. The other questions stay in this quiz.")
        saved=st.form_submit_button("Save review",type="primary")
    if saved:
        try:
            rejected_count=sum(x["review_status"]=="rejected" for x in updated)
            updated=[x for x in updated if x["_keep"] and x["review_status"]!="rejected"]
            if action.startswith("Approve") and any(x["review_status"]!="approved" for x in updated):
                st.error("Mark each remaining question approved before publishing.");return
            updated=[{k:v for k,v in x.items() if k!="_keep"} for x in updated]
            if not updated:st.error("Keep at least one non-rejected question in the quiz.");return
            result=request("PUT",f"/quizzes/{q['id']}",{"title":title,"questions":updated,"status":"published" if action.startswith("Approve") else "draft"})
            st.session_state.review_quiz=result
            message="Quiz saved as a draft." if result["status"]=="draft" else "Quiz approved and published."
            if rejected_count:message+=f" Removed {rejected_count} rejected question{'s' if rejected_count!=1 else ''}; kept the remaining questions."
            st.success(message)
        except Exception as exc:st.error(str(exc))
    if st.button("＋ Add manually written question"):
        q["questions"].append({"id":"manual-"+str(len(q["questions"])+1),"type":"mcq","question":"","options":["Option A","Option B","Option C","Option D"],"answer":"Option A","explanation":"","source":"Instructor-created question","difficulty":"Medium","review_status":"pending","quality":{"needs_review":True}})
        st.session_state.review_quiz=q;st.rerun()


def page_take():
    page_shell("Take a quiz","ASSESSMENT","Answer questions, then submit for server-calculated scoring.")
    quizzes=request("GET","/quizzes"); published=[q for q in quizzes if q["status"]=="published"]
    if not published:
        drafts=[q for q in quizzes if q["status"]!="published"]
        st.info("Your saved quizzes are drafts. Review the questions and publish one before taking it.")
        if drafts:
            st.subheader("Drafts to review")
            labels={f"{q['title']} · {len(q['questions'])} questions":q for q in drafts}
            label=st.selectbox("Choose a draft",list(labels),key="draft_to_review")
            if st.button("Review and publish this quiz",type="primary"):
                st.session_state.review_quiz=request("GET",f"/quizzes/{labels[label]['id']}")
                st.switch_page("pages/4_Review_Questions.py")
        else:
            st.caption("Generate a quiz from your course material first, then review it here.")
            if st.button("Go to Generate Quiz",type="primary"):
                st.switch_page("pages/3_Generate_Quiz.py")
        return
    labels={f"{q['title']} · {len(q['questions'])} questions":q for q in published}; label=st.selectbox("Choose a quiz",list(labels));quiz=labels[label]
    with st.form("take_quiz"):
        answers={}
        for i,q in enumerate(quiz["questions"]):
            answers[q["id"]]=st.radio(f"{i+1}. {q['question']}",q.get("options",["True","False"]),index=None,key=f"take_{quiz['id']}_{i}")
        submitted=st.form_submit_button("Submit answers",type="primary")
    if submitted:
        try:
            result=request("POST",f"/quizzes/{quiz['id']}/submit",{"answers":answers});st.session_state.last_result=result;st.success(f"Score: {result['score']:.1f}% · {result['correct']} of {result['total']} correct.")
        except Exception as exc:st.error(str(exc))


def page_results():
    page_shell("Results and answer review","RESULTS","Review scoring and source explanations for your latest attempt.")
    r=st.session_state.get("last_result")
    if not r:
        attempts=request("GET","/attempts")
        if not attempts:st.info("Complete a quiz to see results here.");return
        labels={f"{a['title']} · {a['score']:.1f}% · {a['created_at'][:19]}":a for a in attempts};label=st.selectbox("Review a saved attempt",list(labels))
        if st.button("Load attempt"):r=request("GET",f"/attempts/{labels[label]['id']}");st.session_state.last_result=r;st.rerun()
        return
    st.metric("Score",f"{r['score']:.1f}%",f"{r['correct']} / {r['total']} correct")
    for i,item in enumerate(r["results"],1):
        st.markdown(f"**{i}. {item['question']['question']}**")
        st.write(("✅ Correct" if item["correct"] else "❌ Review")+f" · Your answer: {item.get('chosen') or 'Unanswered'} · Correct answer: {item['question']['answer']}")
        st.caption(item["question"].get("explanation", ""));st.write("Source excerpt:");st.code(item["question"].get("source", ""))


def page_history():
    page_shell("Quiz history","LIBRARY","Saved assessments and workspace activity.")
    attempts=request("GET","/attempts")
    st.subheader("Recent attempts")
    if attempts:
        for attempt in attempts:
            stamp=str(attempt.get("created_at", ""))[:19].replace("T", " ")
            with st.expander(f"{attempt['title']} · {attempt['score']:.1f}% · {stamp}"):
                st.metric("Score",f"{attempt['score']:.1f}%",f"{attempt['total']} questions")
                if st.button("Review answers",key="attempt_"+attempt["id"]):
                    st.session_state.last_result=request("GET",f"/attempts/{attempt['id']}")
                    st.switch_page("pages/6_Results.py")
    else:st.info("No quiz attempts yet. Published quizzes you complete will appear here.")
    st.divider()
    st.subheader("Saved quizzes")
    quizzes=request("GET","/quizzes")
    if not quizzes:st.info("No quizzes saved yet.");return
    for q in quizzes:
        with st.expander(f"{q['title']} · {len(q['questions'])} questions · {q['status'].title()}"):
            settings=q.get("settings") or {};retrieval=settings.get("retrieval") or {}
            st.write("Created:",q.get("created_at",""));st.write("Generator:",settings.get("generator","rules"));st.write("Retrieval:",retrieval.get("method","not topic-filtered"))
            analytics=request("GET",f"/quizzes/{q['id']}/analytics");st.metric("Attempts",analytics["attempts"],f"Average {analytics['average_score']:.1f}%" if analytics["attempts"] else "No attempts")
            if st.button("Review this quiz",key="history_"+q["id"]):st.session_state.review_quiz=q;st.switch_page("pages/4_Review_Questions.py")


def page_evaluation():
    page_shell("RAG and model evaluation","EVALUATION","Import authorized reference data, inspect retrieval indexes, and review honest metric definitions.")
    status=request("GET","/rag/status");st.subheader("Document retrieval indexes")
    if status["indexes"]:st.dataframe(status["indexes"],use_container_width=True,hide_index=True)
    else:st.info("No indexes built. Upload material to split and index its chunks.")
    docs=request("GET","/documents")
    if docs and st.button("Rebuild all document indexes"):
        with st.spinner("Rebuilding indexes…"):rebuilt=request("POST","/rag/rebuild",{})
        st.success(f"Rebuilt {rebuilt['rebuilt']} indexes.");st.rerun()
    st.subheader("Educational reference dataset")
    uploaded=st.file_uploader("Import JSON array, JSONL, or CSV",type=["json","jsonl","csv"],key="dataset_file",help="Optional reference MCQ data for evaluation and rule/model comparison. This is not needed to upload course material or generate quizzes.")
    st.caption("Use a dataset you are authorized to use. Include its license and source details; expected records contain a passage and a human-written question.")
    c1,c2,c3=st.columns(3);name=c1.text_input("Dataset name");license_name=c2.text_input("License / terms");split=c3.selectbox("Split",["train","validation","test","all"])
    if st.button("Import dataset",disabled=not uploaded):
        try:
            content=uploaded.getvalue().decode("utf-8-sig")
            if uploaded.name.lower().endswith(".csv"):
                import csv,io
                records=list(csv.DictReader(io.StringIO(content)))
                for row in records:
                    if isinstance(row.get("options"),str):row["options"]=json.loads(row["options"])
            else:records=content
            result=request("POST","/dataset/import",{"dataset_name":name,"license":license_name,"split":split,"records":records})
            st.success(f"Imported {result['imported']} records.")
        except Exception as exc:st.error(str(exc))
    dataset=request("GET","/dataset");st.write(f"Imported records: **{dataset['records']}**")
    if dataset["datasets"]:st.dataframe(dataset["datasets"],use_container_width=True,hide_index=True)
    st.subheader("Metrics and limits")
    metrics=request("GET","/metrics");st.json(metrics["automatic"])
    st.caption("Retrieval precision/recall need labeled relevant chunks. Human acceptance needs recorded reviewer labels. Workspace screens do not imply model accuracy. Use ml/scripts/evaluate.py for a same-held-out rule/model comparison.")


def page_settings():
    page_shell("Model configuration","SETTINGS","Configure local retrieval and generation models. No paid API or hard-coded secret is used.")
    active=request("GET","/settings")
    embed=st.text_input("Sentence Transformer embedding model",active["embedding_model"],help="Converts topics and source chunks into vectors for semantic search. This model retrieves passages; it does not write questions.")
    generator=st.selectbox("Question generator",["rules","model"],index=0 if active["generator"]=="rules" else 1,format_func=lambda x:"Rules-based generator" if x=="rules" else "Pretrained model",help="Rules mode uses only deterministic rule-generated question stems. Pretrained mode uses only the Hugging Face model for question stems; if it cannot produce enough usable questions, generation stops instead of mixing in rule-generated stems.")
    model=st.text_input("Hugging Face text-to-text model",active["model_name"],help="Use google/flan-t5-small for automatic download, or enter the path to a checkpoint already stored on this computer.")
    model_path=Path(model).expanduser()
    if model_path.is_dir() and (model_path/"config.json").is_file() and (model_path/"model.safetensors").is_file():
        if generator=="model":st.success(f"Local checkpoint ready: {model_path}. No upload is needed on this computer.")
        else:st.info("The local FLAN-T5 checkpoint is available but inactive. The rules-based generator is selected.")
    local=st.checkbox("Use local files only",value=active["model_local_files_only"])
    if generator=="model" and model.strip()=="google/flan-t5-small" and not local:
        st.info("No local checkpoint is selected. The public model will download automatically the first time generation runs; no ZIP upload is needed.")
    threshold=st.slider("Minimum retrieval score",0.0,1.0,float(active["rag_min_score"]),0.01,help="Minimum cosine-similarity score for a source chunk to be used. Lower values return more loosely related passages; higher values are stricter. This is a search cutoff, not a probability or accuracy score.")
    if st.button("Save settings",type="primary"):
        result=request("POST","/settings",{"embedding_model":embed.strip(),"generator":generator,"model_name":model.strip(),"model_local_files_only":local,"rag_min_score":threshold})
        st.success(f"Saved settings: {result['generator']} generator. This choice will remain active after navigation and app restarts. Rebuild indexes after changing the embedding model.")
    st.code("Install UI: pip install -r requirements.txt\nInstall RAG: pip install -r requirements-rag.txt\nOptional generation: pip install -r requirements-ml.txt",language="text")
    st.caption("Local mode uses the checkpoint stored in this project. On a fresh deployment without that folder, the public Hugging Face model downloads automatically on first generation; the deployment needs internet access and enough memory.")


PAGES={"Dashboard":page_dashboard,"Upload Documents":page_upload,"Generate Quiz":page_generate,"Question Review":page_review,"Take Quiz":page_take,"Results":page_results,"Quiz History":page_history,"RAG & Model Evaluation":page_evaluation,"Settings":page_settings}


def render_page(name):
    try:PAGES[name]()
    except Exception as exc:st.error(f"Could not load this page: {exc}")

