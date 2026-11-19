"""Selection, diagnosis boundaries, durable local data, and visitor isolation."""
import asyncio
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import time
import unittest

from aiohttp import CookieJar
from aiohttp.test_utils import TestClient, TestServer
from assistant_pipeline import AuditedAssistant
from local_store import LocalStore
from review_models import parse_review
from server import create_app
from test_assistant_pipeline import MainModel

CONFIG = json.loads((Path(__file__).parent / "legacy_assistant_config.json").read_text())


class ReviewParsingTests(unittest.TestCase):
    def test_observed_tagged_fusion_output_discards_all_revision_text(self):
        diagnosis = '<Identity>: No problematic cue. <Action>: Thanking a colleague. <SocialState>: Professional courtesy.'
        for marker in ('<RevisedReply>:', '<Revised Reply>:', '####Revised Reply', '###Revision:',
                       '<MorallyAcceptableReply>:', '<Morally Appropriate Reply>:', '<CorrectedResponse>:'):
            parsed = parse_review({'output':diagnosis+' '+marker+' Replace the reply with this text.'},'fusion')
            self.assertEqual(parsed['diagnosis'],diagnosis)
            self.assertNotIn('Replace the reply',parsed['diagnosis'])

    def test_fusion_retains_actions_consequences_foundations_and_judgment(self):
        raw = '(1) Wording. (2) Actions. (3) Consequences. (4) Foundations. (5) Judgment. (6) Rewrite this.'
        diagnosis = parse_review({"output": raw}, "fusion")["diagnosis"]
        self.assertEqual(diagnosis, 'Wording. (2) Actions. (3) Consequences. (4) Foundations. (5) Judgment.')
        self.assertEqual(parse_review({"output": raw}, "light")["diagnosis"], 'Wording.')

    def test_partial_diagnosis_is_rejected_but_complete_boundary_is_usable(self):
        with self.assertRaises(ValueError):
            parse_review({"output": "(1) Wording. (2) Actions unfinished", "truncated": True}, "fusion")
        result = parse_review({"output": "(1) Wording. (5) Judgment. (6)", "truncated": True}, "fusion")
        self.assertIn("(5) Judgment.", result["diagnosis"])

    def test_selected_reviewer_is_used_for_one_rewrite(self):
        class Reviewers:
            def diagnose(self, question, answer, mode):
                self.selected = mode
                return {"diagnosis": f"{mode} diagnosis", "raw_output": "private correction text", "active_adapter": mode}
        for mode in ("light", "fusion"):
            reviewers = Reviewers()
            main = MainModel(["Draft", "Final"])
            result = AuditedAssistant(CONFIG, main=main, reviewers=reviewers).generate({"question": "Question", "review_mode": mode})
            self.assertEqual(reviewers.selected, mode)
            self.assertEqual(result["review_adapter"], mode)
            self.assertIn(f"{mode} diagnosis", main.calls[1][0])
            self.assertNotIn("private correction text", main.calls[1][0])
            self.assertEqual(len(main.calls), 2)


class Engine:
    def load(self):
        pass

    def generate(self, values, progress):
        progress("diagnosing")
        return {"final_answer": "Local answer", "diagnosis": "Local review", "review_mode": values["review_mode"]}


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.database = Path(self.temp.name) / "private" / "requests.sqlite3"
        self.config = copy.deepcopy(CONFIG)
        self.config['storage']['database'] = str(self.database)
        self.client = await self.start_client()

    async def start_client(self, cookies=None):
        client = TestClient(TestServer(create_app(self.config, Engine())), cookie_jar=CookieJar(unsafe=True))
        await client.start_server()
        if cookies:
            client.session.cookie_jar.update_cookies(cookies)
        for _ in range(50):
            info = await (await client.get('/api/info')).json()
            if info['status']=='ready':
                return client
            await asyncio.sleep(.01)
        self.fail('Stub did not become ready')

    async def asyncTearDown(self):
        await self.client.close()
        self.temp.cleanup()

    async def test_modes_are_persisted_and_recoverable_after_restart(self):
        keys=[]
        for mode in ('light','fusion'):
            response = await self.client.post('/api/generate',json={'question':'Keep this local.','review_mode':mode})
            self.assertEqual(response.status,202)
            key=(await response.json())['id']; keys.append(key)
            for _ in range(50):
                result=await (await self.client.get('/api/results/'+key)).json()
                if result['status']=='complete':break
                await asyncio.sleep(.01)
            self.assertEqual(result['result']['review_mode'],mode)
            self.assertNotIn('owner',result)
            self.assertNotIn('values',result)
        cookies=self.client.session.cookie_jar.filter_cookies(self.client.make_url('/'))
        await self.client.close()
        self.client=await self.start_client(cookies)
        result=await (await self.client.get('/api/results/'+keys[-1])).json()
        self.assertEqual(result['result']['review_mode'],'fusion')
        with sqlite3.connect(self.database) as connection:
            rows=connection.execute('SELECT payload FROM jobs').fetchall()
        self.assertEqual([json.loads(r[0])['values']['review_mode'] for r in rows],['light','fusion'])
        self.assertEqual(self.database.stat().st_mode & 0o777,0o600)

    async def test_other_visitors_cannot_read_a_known_job_id(self):
        response=await self.client.post('/api/generate',json={'question':'Hello','review_mode':'fusion'})
        key=(await response.json())['id']
        from aiohttp import ClientSession
        async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as stranger:
            result=await stranger.get(self.client.make_url('/api/results/'+key))
            self.assertEqual(result.status,404)
        for path in ('/data/requests.sqlite3','/data/second-look.sqlite3','/assistant_config.json','/run/server.log','/models/Muse-Glimmer-30B/config.json'):
            self.assertEqual((await self.client.get(path)).status,404)

    async def test_invalid_review_mode_rejected_without_storage(self):
        for mode in ('unknown','../fusion',123,['light']):
            result=await self.client.post('/api/generate',json={'question':'Hello','review_mode':mode})
            self.assertEqual(result.status,400)
        with sqlite3.connect(self.database) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],0)


class RecoveryTests(unittest.TestCase):
    def test_interrupted_work_is_recorded_as_error_on_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data'/'requests.db'
            store=LocalStore(path)
            store.put('one',{'status':'running','created':time.time(),'values':{'question':'Question'}})
            store.close()
            store=LocalStore(path)
            self.assertEqual(store.get('one')['status'],'error')
            self.assertIn('restarted',store.get('one')['error'])
            store.close()


if __name__=='__main__':
    unittest.main()
