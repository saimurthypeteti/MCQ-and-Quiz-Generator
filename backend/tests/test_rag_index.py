import unittest
from pathlib import Path
import uuid

from src.rag_index import index_document, retrieve, split_chunks


class RagIndexTests(unittest.TestCase):
    def setUp(self):
        self.root=Path(__file__).resolve().parent/("_rag_testdata_"+uuid.uuid4().hex)
        self.root.mkdir()
        self.db=self.root/"test.sqlite3"
        self.index_dir=self.root/"indexes"

    def tearDown(self):
        index_dir=self.root/"indexes"
        if index_dir.exists():
            for item in index_dir.iterdir():item.unlink(missing_ok=True)
            index_dir.rmdir()
        if self.db.exists():self.db.unlink()
        self.root.rmdir()

    def test_chunks_preserve_section_and_overlap(self):
        text=("Photosynthesis converts light energy into chemical energy in green plants. "
              "Chlorophyll absorbs light in the leaves and transfers energy. "
              "Plants use water and carbon dioxide to produce glucose and oxygen.")
        chunks=split_chunks([{"label":"Lesson 2","page":4,"text":text}],max_chars=100,overlap_sentences=1)
        self.assertGreaterEqual(len(chunks),2)
        self.assertTrue(all(c["section"]=="Lesson 2" and c["page"]==4 and c["chunk_id"] for c in chunks))
        self.assertEqual(chunks[0]["text"].split(". ")[-1].rstrip("."),chunks[1]["text"].split(". ")[0].rstrip("."))

    def test_persistent_index_maps_retrieval_to_original_chunk(self):
        sections=[{"label":"Photosynthesis","page":3,"text":"Photosynthesis converts sunlight into chemical energy. Chlorophyll absorbs light energy in the leaves of plants."},
                  {"label":"Cell structure","page":8,"text":"The cell membrane controls which substances enter and leave the cell. Mitochondria release energy during cellular respiration."}]
        result=index_document("doc-1",sections,self.db,self.index_dir)
        self.assertIn(result["status"],{"faiss_semantic","keyword_only"})
        matches,meta=retrieve("doc-1","chlorophyll absorbs light energy",self.db,top_k=2)
        self.assertTrue(matches)
        self.assertIn("Chlorophyll",matches[0]["text"])
        self.assertEqual(matches[0]["section"],"Photosynthesis")
        self.assertEqual(matches[0]["page"],3)
        self.assertTrue(matches[0]["chunk_id"])
        self.assertGreaterEqual(meta["chunks_found"],1)
        if result["status"]=="faiss_semantic":
            self.assertTrue(Path(result["vector_path"]).is_file())
            self.assertTrue(meta["method"].startswith("faiss_semantic:"))


if __name__=="__main__":unittest.main()
