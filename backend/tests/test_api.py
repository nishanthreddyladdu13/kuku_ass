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
    def test_animation_timeline_validates_targets_and_continuity(self):
        obj=self.example['objects'][0]
        shot=self.example['shots'][0]
        end=[obj['position'][0]+0.25,obj['position'][1],obj['position'][2]]
        self.example['shots'][0].update(start_time=1,duration=3)
        self.example['animations']=[
            {'id':'move-character','object_id':obj['id'],'type':'move','start_time':0,'duration':2,'from_position':obj['position'],'to_position':end},
            {'id':'camera-zoom','target_type':'camera','shot_id':shot['id'],'type':'zoom','start_time':0,'duration':2,'from_fov':shot['fov'],'to_fov':shot['fov']+5},
        ]
        response=self.client.post('/api/validate',json=self.example)
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['scene']['animations'][0]['object_id'],obj['id'])
        self.assertEqual(response.json()['scene']['shots'][0]['start_time'],1)
    def test_scene_prompt_uses_animations_as_single_movement_source(self):
        self.assertIn('Use scene.animations as the single source of truth for animated movement.',main.SYSTEM)
        self.assertIn('For any object with a move animation, set object.motion to null.',main.SYSTEM)
        self.assertIn('Represent each movement exactly once.',main.SYSTEM)
    def test_exact_duplicate_legacy_motion_is_normalized_to_animation(self):
        obj=self.example['objects'][0]
        start=obj['position']
        end=[start[0]+0.2,start[1],start[2]]
        obj['motion']={'end_position':end,'duration':4}
        self.example['animations']=[{'id':'move-character','object_id':obj['id'],'type':'move','start_time':0,'duration':4,'from_position':start,'to_position':end}]
        with patch.object(main,'invoke',new=AsyncMock(return_value=json.dumps(self.example))) as mock:
            response=self.client.post('/api/scene',json={'script':'A person moves a short distance across the room.'})
        self.assertEqual(response.status_code,200,response.text)
        scene=response.json()['scene']
        normalized=next(item for item in scene['objects'] if item['id']==obj['id'])
        self.assertIsNone(normalized['motion'])
        self.assertEqual(scene['animations'][0]['type'],'move')
        self.assertEqual(mock.await_count,1)
        record=json.loads(next(Path(self.temp.name).glob('*.json')).read_text())
        self.assertEqual(record['attempts'][0]['normalizations'][0]['type'],'duplicate_legacy_motion_removed')
    def test_nonidentical_legacy_motion_is_not_normalized(self):
        obj=self.example['objects'][0]
        start=obj['position']
        motion_end=[start[0]+0.2,start[1],start[2]]
        animation_end=[start[0]+0.4,start[1],start[2]]
        obj['motion']={'end_position':motion_end,'duration':4}
        self.example['animations']=[{'id':'move-character','object_id':obj['id'],'type':'move','start_time':0,'duration':4,'from_position':start,'to_position':animation_end}]
        raw=json.dumps(self.example)
        candidate,normalized=main.normalize_duplicate_legacy_motions(raw)
        self.assertEqual(candidate,raw)
        self.assertEqual(normalized,[])
    def test_animation_rejects_missing_target_or_discontinuous_start(self):
        obj=self.example['objects'][0]
        self.example['animations']=[{'id':'bad-move','object_id':'missing','type':'move','start_time':0,'duration':2,'from_position':obj['position'],'to_position':[obj['position'][0]+0.2,obj['position'][1],obj['position'][2]]}]
        self.assertEqual(self.client.post('/api/validate',json=self.example).status_code,422)
        self.example['animations'][0]['object_id']=obj['id']
        self.example['animations'][0]['from_position']=[obj['position'][0]+1,obj['position'][1],obj['position'][2]]
        self.assertEqual(self.client.post('/api/validate',json=self.example).status_code,422)
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
    def test_scene_image_uses_configured_model_and_returns_image(self):
        image_data='aGVsbG8='
        async def fake_invoke(prompt,model,request_log,reference_image=None):
            request_log.update(effective_model='core',image_bytes=5)
            return 'image/png',image_data
        with patch.dict(os.environ,{'STABILITY_API_KEY':'test-key','IMAGE_MODEL':'core'}),patch.object(main,'invoke_image',new=AsyncMock(side_effect=fake_invoke)) as mock:
            r=self.client.post('/api/scene-image',json={'scene':self.example,'shot_id':self.example['shots'][0]['id']})
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(r.json()['model'],'core')
        self.assertEqual(r.json()['mime_type'],'image/png')
        self.assertEqual(r.json()['data_base64'],image_data)
        self.assertEqual(r.json()['provider'],'stability')
        self.assertEqual(r.json()['shot_id'],self.example['shots'][0]['id'])
        prompt=mock.await_args.args[0]
        self.assertIn(self.example['title'],prompt)
        self.assertIn(self.example['shots'][0]['name'],prompt)
        self.assertIn(self.example['shots'][0]['action'],prompt)
        self.assertNotIn('test-key',r.text)
    def test_scene_image_reports_upstream_503_without_retry(self):
        request=httpx.Request('POST','https://api.stability.ai/v2beta/stable-image/generate/core')
        response=httpx.Response(503,json={'message':'Image service is busy.'},request=request)
        error=httpx.HTTPStatusError('upstream error',request=request,response=response)
        with patch.dict(os.environ,{'STABILITY_API_KEY':'test-key','IMAGE_MODEL':'core'}),patch.object(main,'invoke_image',new=AsyncMock(side_effect=error)) as mock:
            r=self.client.post('/api/scene-image',json={'scene':self.example,'shot_id':self.example['shots'][0]['id']})
        self.assertEqual(r.status_code,503)
        self.assertIn('Image service is busy',r.json()['detail'])
        self.assertEqual(mock.await_count,1)
    def test_scene_image_surfaces_upstream_quota_429(self):
        request=httpx.Request('POST','https://api.stability.ai/v2beta/stable-image/generate/core')
        response=httpx.Response(429,json={'message':'Daily image quota exceeded.'},request=request)
        error=httpx.HTTPStatusError('upstream error',request=request,response=response)
        with patch.dict(os.environ,{'STABILITY_API_KEY':'test-key','IMAGE_MODEL':'core'}),patch.object(main,'invoke_image',new=AsyncMock(side_effect=error)):
            r=self.client.post('/api/scene-image',json={'scene':self.example,'shot_id':self.example['shots'][0]['id']})
        self.assertEqual(r.status_code,429)
        self.assertIn('Daily image quota exceeded',r.json()['detail'])
        self.assertNotIn('test-key',r.text)
    def test_scene_image_reports_timeout(self):
        with patch.dict(os.environ,{'STABILITY_API_KEY':'test-key','IMAGE_MODEL':'core'}),patch.object(main,'invoke_image',new=AsyncMock(side_effect=httpx.ReadTimeout('timed out'))) as mock:
            r=self.client.post('/api/scene-image',json={'scene':self.example,'shot_id':self.example['shots'][0]['id']})
        self.assertEqual(r.status_code,504)
        self.assertIn('timed out',r.json()['detail'])
        self.assertEqual(mock.await_count,1)
    def test_scene_image_rejects_unknown_shot_before_provider_call(self):
        with patch.object(main,'invoke_image',new=AsyncMock()) as mock:
            r=self.client.post('/api/scene-image',json={'scene':self.example,'shot_id':'missing-shot'})
        self.assertEqual(r.status_code,404)
        mock.assert_not_awaited()
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
