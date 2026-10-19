import json, os, threading, unittest, urllib.request, uuid
from io import BytesIO
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app

PASSAGE="Photosynthesis is the process by which green plants convert sunlight into chemical energy. Chlorophyll absorbs light energy in the leaves. The plant uses carbon dioxide and water to produce glucose and oxygen."

class CoreTests(unittest.TestCase):
    def test_cleaning_and_txt_extraction(self):
        text,sections=app.extract("lesson.txt",PASSAGE.encode())
        self.assertIn("Photosynthesis",text); self.assertEqual(sections[0]["label"],"Section 1")
    def test_docx_extraction(self):
        from docx import Document
        buffer=BytesIO();document=Document();document.add_heading("Lesson",level=1);document.add_paragraph(PASSAGE);document.save(buffer)
        text,sections=app.extract("lesson.docx",buffer.getvalue())
        self.assertIn("Photosynthesis",text);self.assertTrue(sections)
    def test_pdf_extraction_preserves_page_number(self):
        import pymupdf
        buffer=BytesIO();document=pymupdf.open();page=document.new_page();page.insert_text((72,72),PASSAGE);buffer.write(document.tobytes());document.close()
        text,sections=app.extract("lesson.pdf",buffer.getvalue())
        self.assertIn("Photosynthesis",text);self.assertEqual(sections[0]["page"],1);self.assertEqual(sections[0]["label"],"Page 1")
    def test_empty_and_unsupported(self):
        with self.assertRaises(ValueError):app.extract("bad.exe",b"x")
        with self.assertRaises(ValueError):app.extract("empty.txt",b"")
    def test_question_contains_source_answer_and_review_flag(self):
        q=app.make_question("Chlorophyll absorbs light energy in the leaves.")
        self.assertIn(q["answer"].lower(),q["source"].lower()); self.assertTrue(q["quality"]["needs_review"])
    def test_pretrained_mode_does_not_mix_in_rule_question_stems(self):
        class BadModel:
            def __call__(self,*args,**kwargs):return [{"generated_text":"unusable output"}]
        previous_pipeline=app.MODEL_PIPELINE;previous_name=os.environ.get("MODEL_NAME")
        app.MODEL_PIPELINE=BadModel();os.environ["MODEL_NAME"]="test-checkpoint"
        try:
            with self.assertRaises(app.ModelGenerationError) as error:
                app.model_questions(["Chlorophyll absorbs light energy in the leaves."],1,"Easy","mcq",4)
            self.assertIn("No rules-based questions were added",str(error.exception))
        finally:
            app.MODEL_PIPELINE=previous_pipeline
            if previous_name is None:os.environ.pop("MODEL_NAME",None)
            else:os.environ["MODEL_NAME"]=previous_name
    def test_dataset_split_prevents_passage_leakage(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"ml"/"scripts"))
        from prepare_dataset import prepare
        from evaluate import metric_summary,rule_generate,valid
        rows=[]
        for i in range(12):rows.append({"passage_id":f"p{i//2}","passage":f"Passage {i//2}","question":f"Question {i}","options":["a","b"],"correct_answer":"a"})
        splits,manifest=prepare(rows,seed=7)
        ids=[{r["passage_id"] for r in batch} for batch in splits.values()]
        self.assertFalse(ids[0]&ids[1]);self.assertFalse(ids[0]&ids[2]);self.assertFalse(ids[1]&ids[2]);self.assertEqual(manifest["valid_records"],12)
        baseline=rule_generate(PASSAGE);self.assertTrue(valid(baseline))
        reference={"question":"What does photosynthesis produce?","passage":PASSAGE}
        self.assertEqual(metric_summary([baseline],[reference])["semantic_similarity_to_human_question"],None)
    def test_sciq_adapter_preserves_question_answer_and_support(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"ml"/"scripts"))
        from prepare_dataset import prepare
        raw={"id":"sample-1","question":"What do green plants produce?","correct_answer":"glucose","distractor1":"water","distractor2":"soil","distractor3":"light","support":PASSAGE}
        splits,manifest=prepare([raw],seed=1)
        prepared=[item for batch in splits.values() for item in batch]
        self.assertEqual(manifest["valid_records"],1);self.assertEqual(prepared[0]["correct_answer"],"glucose");self.assertEqual(prepared[0]["passage"],PASSAGE)

class HTTPFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir=Path(__file__).resolve().parent/("_testdata_"+uuid.uuid4().hex);cls.data_dir.mkdir(exist_ok=True);os.environ["MCQ_DATA_DIR"]=str(cls.data_dir);app.DATA_DIR=cls.data_dir;app.DB_PATH=app.DATA_DIR/"mcq.sqlite3";app.init_db()
        cls.server=app.ThreadingHTTPServer(("127.0.0.1",0),app.API);cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True);cls.thread.start();cls.base=f"http://127.0.0.1:{cls.server.server_port}"
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown();cls.server.server_close();cls.thread.join(timeout=2)
        for child in cls.data_dir.iterdir():
            if child.is_dir():
                for nested in child.iterdir():nested.unlink(missing_ok=True)
                child.rmdir()
            else:child.unlink(missing_ok=True)
        cls.data_dir.rmdir()
    def request(self,path,method="GET",data=None):
        req=urllib.request.Request(self.base+path,data=json.dumps(data).encode() if data is not None else None,headers={"Content-Type":"application/json"} if data is not None else {},method=method)
        with urllib.request.urlopen(req) as response:return response.status,json.loads(response.read())
    def test_generate_save_submit_and_analytics(self):
        status,doc=self.request("/api/documents/text","POST",{"name":"lesson.txt","text":PASSAGE});self.assertEqual(status,201)
        status,quiz=self.request("/api/generate","POST",{"document_id":doc["id"],"count":3});self.assertEqual(status,201);self.assertTrue(quiz["questions"])
        _,validation=self.request("/api/validation","POST",{"questions":quiz["questions"]});self.assertEqual(validation["summary"]["count"],len(quiz["questions"]));self.assertTrue(validation["summary"]["requires_human_review"])
        status,saved=self.request("/api/quizzes/"+quiz["id"],"PUT",{"questions":quiz["questions"],"status":"published"});self.assertEqual(saved["status"],"published")
        ans={q["id"]:q["answer"] for q in saved["questions"]};status,result=self.request("/api/quizzes/"+quiz["id"]+"/submit","POST",{"answers":ans});self.assertEqual(result["score"],100.0)
        _,dash=self.request("/api/dashboard");self.assertEqual(dash["documents"],1);self.assertEqual(dash["attempts"],1)
    def test_topic_generation_uses_persisted_retrieval_and_source_mapping(self):
        _,doc=self.request("/api/documents/text","POST",{"name":"biology.txt","text":PASSAGE+"\n\nThe cell membrane controls which substances enter and leave the cell."})
        status,quiz=self.request("/api/generate","POST",{"document_id":doc["id"],"topic":"chlorophyll absorbs light energy","count":2})
        self.assertEqual(status,201);self.assertGreater(quiz["retrieval"]["chunks_found"],0)
        self.assertTrue(all(q.get("topic")=="chlorophyll absorbs light energy" for q in quiz["questions"]))
        self.assertTrue(all(q.get("source_chunk_id") and q.get("retrieval_score") is not None for q in quiz["questions"]))
        _,status_data=self.request("/api/rag/status");self.assertTrue(any(x["document_id"]==doc["id"] for x in status_data["indexes"]))
    def test_model_unavailable_is_explicit_and_static_app_is_served(self):
        with urllib.request.urlopen(self.base+"/") as response:self.assertIn("id=\"view\"",response.read().decode())
        previous=os.environ.get("GENERATOR");previous_model=os.environ.get("MODEL_NAME");os.environ["GENERATOR"]="model";os.environ.pop("MODEL_NAME",None)
        try:
            _,doc=self.request("/api/documents/text","POST",{"name":"lesson.txt","text":PASSAGE})
            req=urllib.request.Request(self.base+"/api/generate",data=json.dumps({"document_id":doc["id"]}).encode(),headers={"Content-Type":"application/json"},method="POST")
            with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(req)
            self.assertEqual(error.exception.code,503)
        finally:
            if previous is None:os.environ.pop("GENERATOR",None)
            else:os.environ["GENERATOR"]=previous
            if previous_model is None:os.environ.pop("MODEL_NAME",None)
            else:os.environ["MODEL_NAME"]=previous_model
    def test_generator_setting_survives_app_reinitialization(self):
        keys=("GENERATOR","MODEL_NAME","MODEL_LOCAL_FILES_ONLY","EMBEDDING_MODEL","RAG_MIN_SCORE")
        previous={key:os.environ.get(key) for key in keys}
        try:
            _,saved=self.request("/api/settings","POST",{"generator":"rules","model_name":"","model_local_files_only":True,"embedding_model":"test-embeddings","rag_min_score":0.2})
            self.assertEqual(saved["generator"],"rules")
            os.environ["GENERATOR"]="model";os.environ["MODEL_NAME"]="google/flan-t5-small"
            app.init_db()
            _,active=self.request("/api/settings")
            self.assertEqual(active["generator"],"rules")
            self.assertEqual(active["model_name"],"")
        finally:
            for key,value in previous.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
    def test_multipart_txt_upload_and_filename_sanitizing(self):
        boundary="----mcq-test-boundary"
        body=(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"..\\\\lesson.txt\"\r\nContent-Type: text/plain\r\n\r\n{PASSAGE}\r\n--{boundary}--\r\n").encode()
        request=urllib.request.Request(self.base+"/api/documents",data=body,headers={"Content-Type":f"multipart/form-data; boundary={boundary}"},method="POST")
        with urllib.request.urlopen(request) as response:
            value=json.loads(response.read());self.assertEqual(response.status,201);self.assertEqual(value["name"],"lesson.txt");self.assertIn("Photosynthesis",value["text"])
    def test_import_real_schema_dataset_with_license_metadata(self):
        row={"passage_id":"edu-1","passage":PASSAGE,"question":"What does photosynthesis produce?","options":["Glucose and oxygen","Only water"],"correct_answer":"Glucose and oxygen","annotation_source":"human-written"}
        status,result=self.request("/api/dataset/import","POST",{"dataset_name":"Local authorized QA","license":"CC BY 4.0","split":"test","records":[row]})
        self.assertEqual(status,201);self.assertEqual(result["imported"],1)
        _,dataset=self.request("/api/dataset");self.assertEqual(dataset["records"],1);self.assertEqual(dataset["datasets"][0]["license"],"CC BY 4.0")
        _,records=self.request("/api/dataset/records");self.assertEqual(records[0]["annotation_source"],"human-written")
        with urllib.request.urlopen(self.base+"/api/dataset/export?split=test") as response:
            exported=json.loads(response.read().decode().splitlines()[0]);self.assertEqual(exported["license"],"CC BY 4.0");self.assertEqual(exported["split"],"test")
        _,metrics=self.request("/api/metrics");self.assertEqual(metrics["dataset"]["records"],1)

if __name__=="__main__":unittest.main()
