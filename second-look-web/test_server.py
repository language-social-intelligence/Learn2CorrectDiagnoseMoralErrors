"""HTTP contract tests, independent of the heavyweight model load."""
import asyncio
import json
from pathlib import Path
import unittest

from aiohttp.test_utils import TestClient, TestServer
from server import build_prompt, create_app

CONFIG = json.loads((Path(__file__).parent / "config.json").read_text())


class StubEngine:
    def load(self):
        pass

    def generate(self, values):
        return {"output": "My answer is: benign.", "verdict": "benign", "tokens": 5, "seconds": 0.1, "truncated": False}


class ApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = TestClient(TestServer(create_app(CONFIG, StubEngine())))
        await self.client.start_server()
        for _ in range(30):
            data = await (await self.client.get("/api/info")).json()
            if data["status"] == "ready":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(data["status"], "ready")

    async def asyncTearDown(self):
        await self.client.close()

    async def test_complete_request_and_private_artifacts(self):
        response = await self.client.post("/api/generate", json={"text": "Thank you."})
        self.assertEqual(response.status, 202)
        key = (await response.json())["id"]
        for _ in range(30):
            result = await (await self.client.get("/api/results/" + key)).json()
            if result["status"] == "complete":
                break
            await asyncio.sleep(0.01)
        self.assertEqual(result["result"]["verdict"], "benign")
        for path in ("/config.json", "/server.py", "/run/server.log", "/model.safetensors"):
            self.assertEqual((await self.client.get(path)).status, 404)
        info = await (await self.client.get("/api/info")).json()
        self.assertNotIn("checkpoint", info)

    async def test_invalid_inputs_and_cross_origin(self):
        for value in ({"text": "  "}, {"text": 7}, {"text": "a" * 4001}, [], {"text": "hi", "checkpoint": "/tmp/model"}):
            self.assertEqual((await self.client.post("/api/generate", json=value)).status, 400)
        response = await self.client.post("/api/generate", json={"text": "Hello"}, headers={"Origin": "https://other.example"})
        self.assertEqual(response.status, 403)
        self.assertEqual((await self.client.get("/api/results/missing")).status, 404)

    async def test_prompt_format(self):
        self.assertEqual(build_prompt(CONFIG, {"text": " hi\nthere "}),
                         ' Your task is to determine the given utterance is toxic or not. The utterance is "hi there".###Inference: ')


if __name__ == "__main__":
    unittest.main()
