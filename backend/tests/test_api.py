import unittest, json, tempfile, os
from pathlib import Path
from unittest.mock import patch, AsyncMock
import httpx
from fastapi.testclient import TestClient
from app import main
from app.models import Scene, spatial_warnings

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.data_patch=patch.object(main,'DATA',Path(self.temp.name));self.data_patch.start()
        self.client=TestClient(main.app)
        self.example=json.loads((main.BASE/'examples'/'1.json').read_text())
    def tearDown(self): self.data_patch.stop();self.temp.cleanup()
    def test_examples_are_valid_distinct_scenes(self):
        result=self.client.get('/api/examples').json()
        self.assertEqual(len(result),3)
        self.assertEqual(len({p['title'] for p in result}),3)
        for p in result:
            self.assertEqual(spatial_warnings(Scene.model_validate(p)),[])
    def test_duplicate_ids_rejected(self):
        self.example['objects'][1]['id']=self.example['objects'][0]['id']
        self.assertEqual(self.client.post('/api/validate',json=self.example).status_code,422)
    def test_new_scene_with_single_person_and_bed(self):
        p=self.example;p['objects']=[p['objects'][0],dict(p['objects'][2],kind='bed',shape='rectangular')]
        p['room']={'width':8,'depth':5,'height':3}
        self.assertEqual(self.client.post('/api/validate',json=p).status_code,200)
    def test_missing_credentials_explicit(self):
        with patch.dict(os.environ,{'LLM_API_KEY':'','LLM_MODEL':''}):
            r=self.client.post('/api/scene',json={'script':'A new person beside a counter.'})
            self.assertEqual(r.status_code,503)
    def test_temporary_503_retries_then_succeeds(self):
        request=httpx.Request('POST','https://generativelanguage.googleapis.com/v1beta/models/test:generateContent')
        response=httpx.Response(503,json={'error':{'code':503,'status':'UNAVAILABLE','message':'Model is temporarily busy.'}},request=request)
        error=httpx.HTTPStatusError('upstream error',request=request,response=response)
        with patch.object(main,'invoke',new=AsyncMock(side_effect=[error,json.dumps(self.example)])) as mock, patch.object(main.asyncio,'sleep',new=AsyncMock()):
            r=self.client.post('/api/scene',json={'script':'A person beside a counter.'})
        self.assertEqual(r.status_code,200)
        self.assertEqual(mock.await_count,2)
        record=json.loads(next(Path(self.temp.name).glob('*.json')).read_text())
        self.assertEqual(record['provider_requests'][0]['upstream_http_status'],503)
        self.assertEqual(record['provider_requests'][0]['failure_category'],'upstream_unavailable')
    def test_three_503_attempts_return_informative_error(self):
        request=httpx.Request('POST','https://generativelanguage.googleapis.com/v1beta/models/test:generateContent')
        def unavailable():
            response=httpx.Response(503,json={'error':{'code':503,'status':'UNAVAILABLE','message':'Model is temporarily busy.'}},request=request)
            return httpx.HTTPStatusError('upstream error',request=request,response=response)
        with patch.object(main,'invoke',new=AsyncMock(side_effect=[unavailable(),unavailable(),unavailable()])) as mock, patch.object(main.asyncio,'sleep',new=AsyncMock()):
            r=self.client.post('/api/scene',json={'script':'A person beside a counter.'})
        self.assertEqual(r.status_code,503)
        self.assertIn('after 3 attempt(s)',r.json()['detail'])
        self.assertIn('temporarily busy',r.json()['detail'])
        self.assertEqual(mock.await_count,3)
        record=json.loads(next(Path(self.temp.name).glob('*.json')).read_text())
        self.assertEqual(len(record['provider_requests']),3)
        self.assertTrue(all(item['failure_category']=='upstream_unavailable' for item in record['provider_requests']))
    def test_503_attempt_limit_includes_scene_repair_request(self):
        request=httpx.Request('POST','https://generativelanguage.googleapis.com/v1beta/models/test:generateContent')
        def unavailable():
            response=httpx.Response(503,json={'error':{'code':503,'status':'UNAVAILABLE','message':'Model is temporarily busy.'}},request=request)
            return httpx.HTTPStatusError('upstream error',request=request,response=response)
        with patch.object(main,'invoke',new=AsyncMock(side_effect=['not json',unavailable(),unavailable()])) as mock, patch.object(main.asyncio,'sleep',new=AsyncMock()):
            r=self.client.post('/api/scene',json={'script':'A person beside a counter.'})
        self.assertEqual(r.status_code,503)
        self.assertEqual(mock.await_count,3)
        record=json.loads(next(Path(self.temp.name).glob('*.json')).read_text())
        self.assertEqual(len(record['provider_requests']),3)
    def test_non_503_provider_error_is_not_retried(self):
        request=httpx.Request('POST','https://generativelanguage.googleapis.com/v1beta/models/test:generateContent')
        response=httpx.Response(400,json={'error':{'code':400,'status':'INVALID_ARGUMENT','message':'Invalid request.'}},request=request)
        error=httpx.HTTPStatusError('upstream error',request=request,response=response)
        with patch.object(main,'invoke',new=AsyncMock(side_effect=error)) as mock, patch.object(main.asyncio,'sleep',new=AsyncMock()):
            r=self.client.post('/api/scene',json={'script':'A person beside a counter.'})
        self.assertEqual(r.status_code,400)
        self.assertEqual(mock.await_count,1)
        self.assertEqual(len(json.loads(next(Path(self.temp.name).glob('*.json')).read_text())['provider_requests']),1)
    def test_successful_scene_uses_configured_model_once(self):
        with patch.object(main,'invoke',new=AsyncMock(return_value=json.dumps(self.example))) as mock:
            r=self.client.post('/api/scene',json={'script':'A newly worded conversation in an office.'})
        self.assertEqual(r.status_code,200)
        self.assertEqual(mock.await_count,1)
        record=json.loads(next(Path(self.temp.name).glob('*.json')).read_text())
        self.assertEqual(record['provider_requests'][0]['upstream_http_status'],200)
        self.assertEqual(record['provider_requests'][0]['model'],record['model'])
    def test_gemini_schema_is_inlined_without_unsupported_array_bounds(self):
        response_schema=main.gemini_response_schema()
        def schema_keys(node):
            if isinstance(node,dict):
                for key,value in node.items():
                    yield key
                    yield from schema_keys(value)
            elif isinstance(node,list):
                for value in node: yield from schema_keys(value)
        keys=set(schema_keys(response_schema))
        self.assertFalse(keys & {'$defs','$ref','prefixItems','additionalProperties','exclusiveMinimum','minItems','maxItems','const'})
        self.assertIn('title',response_schema['properties'])
        objects=response_schema['properties']['objects']
        self.assertEqual(objects['description'],'1 to 20 items.')
        self.assertEqual(objects['items']['properties']['position']['description'],'Exactly 3 items.')
    def test_scene_overall_deadline_returns_504(self):
        async def slow_invoke(prompt, request_log=None):
            await main.asyncio.sleep(0.05)
            return json.dumps(self.example)
        with patch.object(main,'invoke',new=AsyncMock(side_effect=slow_invoke)), patch.object(main,'SCENE_RESPONSE_DEADLINE_SECONDS',0.001):
            r=self.client.post('/api/scene',json={'script':'A person beside a counter.'})
        self.assertEqual(r.status_code,504)
        self.assertIn('deadline',r.json()['detail'])
    def test_provider_timeout_is_reported_as_gateway_timeout(self):
        with patch.object(main,'invoke',new=AsyncMock(side_effect=httpx.ReadTimeout('timed out'))):
            r=self.client.post('/api/scene',json={'script':'A person beside a table.'})
        self.assertEqual(r.status_code,504)
        self.assertIn('timed out',r.json()['detail'])
        record=json.loads(next(Path(self.temp.name).glob('*.json')).read_text())
        self.assertEqual(record['error']['type'],'ReadTimeout')
    def test_outside_motion_warns(self):
        self.example['objects'][0]['motion']={'end_position':[40,0,0],'duration':3}
        r=self.client.post('/api/validate',json=self.example)
        self.assertTrue(r.json()['warnings'])
    def test_relation_reference_rejected(self):
        self.example['relations']=[{'subject':'missing','relation':'near','target':'ravi'}]
        self.assertEqual(self.client.post('/api/validate',json=self.example).status_code,422)
    def test_repair_keeps_raw_outputs(self):
        with patch.object(main,'invoke',new=AsyncMock(side_effect=['not json',json.dumps(self.example)])) as mock:
            r=self.client.post('/api/scene',json={'script':'A newly worded conversation in an office.'})
            self.assertEqual(r.status_code,200);self.assertEqual(mock.await_count,2)
        record=json.loads(next(Path(self.temp.name).glob('*.json')).read_text())
        self.assertEqual(record['attempts'][0]['raw'],'not json')
        self.assertEqual(record['scene']['description'],'A newly worded conversation in an office.')
    def test_no_infinite_retry(self):
        with patch.object(main,'invoke',new=AsyncMock(return_value='{}')) as mock:
            r=self.client.post('/api/scene',json={'script':'A person beside a table.'})
            self.assertEqual(r.status_code,422);self.assertEqual(mock.await_count,2)
    def test_invalid_input_never_calls_provider(self):
        with patch.object(main,'invoke',new=AsyncMock()) as mock:
            self.assertEqual(self.client.post('/api/scene',json={'script':'hi'}).status_code,422);mock.assert_not_called()
    def test_facing_relation(self):
        self.example['relations']=[{'subject':'asha','target':'ravi','relation':'faces'}]
        self.assertEqual(spatial_warnings(Scene.model_validate(self.example)),[])
        self.example['objects'][0]['rotation']=0
        self.assertIn('does not face',spatial_warnings(Scene.model_validate(self.example))[0])

if __name__=='__main__':unittest.main()
