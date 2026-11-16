from pathlib import Path
import os, json, uuid, time, asyncio
import logging, random
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError
from .models import Scene, SceneRequest, spatial_warnings

BASE=Path(__file__).resolve().parents[1]
load_dotenv(BASE/'.env')
app=FastAPI(title='CineControl API',version='0.2.0')
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],allow_methods=['GET','POST'],allow_headers=['Content-Type'])
DATA=BASE/'data'/'runs'
SLOTS=asyncio.Semaphore(1)
MAX_PROVIDER_ATTEMPTS=3
PROVIDER_TIMEOUT_SECONDS=20
RETRY_BACKOFF_BASE_SECONDS=0.5
MAX_RETRY_BACKOFF_SECONDS=2
SCENE_RESPONSE_DEADLINE_SECONDS=70
MAX_OUTPUT_TOKENS=4096
logger=logging.getLogger(__name__)
SYSTEM='''You plan simple indoor film sets. Return only JSON matching the supplied schema.
Units metres; +Y up; origin room centre. Object position is bottom centre. rotation is degrees around Y; 0 faces +Z, 180 faces -Z. Dimensions are width,height,depth.
Support one room, 1-4 characters, up to 20 major objects and 4 shots. Objects: character/table/chair/sofa/bed/counter/shelf/door/window/box. Door and window are filled wall markers, not traversable openings. Place them just inside a wall. Only table supports round shape. Character dimensions nominal 0.6,1.8,0.6 (seated height 1.4); seated poses need explicit chairs if appropriate.
Translate the actual script, never substitute an unrelated example. Record unsupported actions (fights, detailed hand contact, crowds) in unsupported. Inferred details go in assumptions. Preserve stated counts, shapes, names and positions. Interpret behind relative to a character's facing and explain it. Keep cameras inside room, not inside objects; frame subject heads. Keep objects and motion endpoints in bounds. Motion means straight-line position interpolation, not realistic walking. Keep path clear; no pathfinding. Left/right relations refer to world X, not screen. Add explicit supported relations. Request beyond scope: return a useful basic layout and report every omitted detail. No arbitrary code.'''

@app.get('/api/health')
def health():
    provider=os.getenv('LLM_PROVIDER','gemini')
    return {'status':'ok','provider':provider,'model':os.getenv('LLM_MODEL',''),'configured':bool(os.getenv('LLM_API_KEY') and os.getenv('LLM_MODEL')),'video_generation':'manual_external'}
@app.get('/api/schema')
def schema(): return Scene.model_json_schema()

def gemini_response_schema()->dict:
    source=Scene.model_json_schema()
    definitions=source.get('$defs',{})

    def normalize(node:dict)->dict:
        reference=node.get('$ref')
        if reference:
            target=definitions[reference.rsplit('/',1)[-1]]
            return normalize({**target,**{key:value for key,value in node.items() if key!='$ref'}})
        if 'anyOf' in node:
            choices=[choice for choice in node['anyOf'] if choice.get('type')!='null']
            if len(choices)==1:
                overlay={key:value for key,value in node.items() if key not in {'anyOf','title','default'}}
                return normalize({**choices[0],**overlay})
        result={}
        for key,value in node.items():
            if key in {'$defs','$ref','title','default','additionalProperties','exclusiveMinimum','minItems','maxItems'}: continue
            if key=='properties' and isinstance(value,dict):
                result[key]={name:normalize(property_schema) for name,property_schema in value.items()}
            elif key=='const':
                result['enum']=[value]
            elif key=='prefixItems':
                common=value[0] if value and all(item==value[0] for item in value) else {'type':value[0].get('type','number')}
                result['items']=normalize(common)
            elif isinstance(value,dict):
                result[key]=normalize(value)
            elif isinstance(value,list):
                result[key]=[normalize(item) if isinstance(item,dict) else item for item in value]
            else:
                result[key]=value
        if node.get('type')=='array':
            minimum=node.get('minItems')
            maximum=node.get('maxItems')
            if minimum is not None and minimum==maximum:
                result.setdefault('description',f'Exactly {minimum} items.')
            elif minimum is not None and maximum is not None:
                result.setdefault('description',f'{minimum} to {maximum} items.')
            elif maximum is not None:
                result.setdefault('description',f'At most {maximum} items.')
        return result

    return normalize(source)

@app.get('/api/examples')
def examples():
    return [json.loads(p.read_text()) for p in sorted((BASE/'examples').glob('*.json'))]
@app.post('/api/validate')
def validate(scene:Scene): return {'scene':scene.model_dump(),'warnings':spatial_warnings(scene)}

async def invoke(prompt:str,request_log:dict|None=None)->str:
    key=os.getenv('LLM_API_KEY');model=os.getenv('LLM_MODEL','').strip()
    if not key or not model: raise HTTPException(503,'Configure LLM_API_KEY and LLM_MODEL in backend/.env, then restart. Example scenes remain available; no automatic fake generation.')
    provider=os.getenv('LLM_PROVIDER','gemini')
    async with httpx.AsyncClient(timeout=httpx.Timeout(PROVIDER_TIMEOUT_SECONDS,connect=5)) as client:
        if provider=='gemini':
            if not all(c.isalnum() or c in '-_.' for c in model): raise HTTPException(500,'Invalid model ID configuration')
            schema_started=time.monotonic()
            scene_schema=gemini_response_schema()
            schema_seconds=time.monotonic()-schema_started
            prepare_started=time.monotonic()
            schema_json=json.dumps(scene_schema,separators=(',',':'))
            payload={'systemInstruction':{'parts':[{'text':SYSTEM}]},'contents':[{'role':'user','parts':[{'text':prompt}]}],'generationConfig':{'temperature':0.1,'maxOutputTokens':MAX_OUTPUT_TOKENS,'responseMimeType':'application/json','responseJsonSchema':scene_schema}}
            serialized_payload=json.dumps(payload,separators=(',',':'),ensure_ascii=False)
            schema_bytes=len(schema_json.encode('utf-8'))
            request_body_bytes=len(serialized_payload.encode('utf-8'))
            prepare_seconds=time.monotonic()-prepare_started
            if request_log is not None:
                request_log.update(schema_generation_seconds=round(schema_seconds,3),request_preparation_seconds=round(prepare_seconds,3),prompt_chars=len(prompt),system_instruction_chars=len(SYSTEM),schema_bytes=schema_bytes,request_body_bytes=request_body_bytes)
            logger.info('LLM stage=request_prepared model=%s schema_seconds=%.3f preparation_seconds=%.3f prompt_chars=%d system_chars=%d schema_bytes=%d request_body_bytes=%d max_output_tokens=%d',model,schema_seconds,prepare_seconds,len(prompt),len(SYSTEM),schema_bytes,request_body_bytes,MAX_OUTPUT_TOKENS)
            request_started=time.monotonic()
            try:
                r=await client.post(f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent',headers={'x-goog-api-key':key},json=payload)
            finally:
                request_seconds=time.monotonic()-request_started
                if request_log is not None: request_log['upstream_request_seconds']=round(request_seconds,3)
            if request_log is not None: request_log['upstream_http_status']=r.status_code
            logger.info('LLM stage=upstream_response model=%s duration_seconds=%.3f upstream_http_status=%s',model,request_seconds,r.status_code)
            r.raise_for_status()
            parse_started=time.monotonic()
            try:
                data=r.json()
                text=''.join(p.get('text','') for p in data['candidates'][0]['content']['parts'])
            finally:
                parse_seconds=time.monotonic()-parse_started
                if request_log is not None: request_log['response_parse_seconds']=round(parse_seconds,3)
            logger.info('LLM stage=response_parsed model=%s duration_seconds=%.3f response_chars=%d',model,parse_seconds,len(text))
            return text
        if provider=='openai_compatible':
            base=os.getenv('LLM_BASE_URL','').rstrip('/')
            if not base.startswith(('https://','http://localhost','http://127.0.0.1')): raise HTTPException(500,'Configure an HTTPS LLM_BASE_URL (or localhost for a local model).')
            r=await client.post(base+'/chat/completions',headers={'Authorization':f'Bearer {key}'},json={'model':model,'messages':[{'role':'system','content':SYSTEM+'\nSchema:\n'+json.dumps(Scene.model_json_schema())},{'role':'user','content':prompt}],'temperature':0.1,'response_format':{'type':'json_object'}})
            r.raise_for_status();return r.json()['choices'][0]['message']['content']
        raise HTTPException(500,'Unsupported LLM_PROVIDER')

async def invoke_with_retries(prompt:str,requests:list[dict],request_budget:dict)->str:
    model=os.getenv('LLM_MODEL','').strip()
    retry_count=0
    while request_budget['used']<MAX_PROVIDER_ATTEMPTS:
        started=time.monotonic()
        request_budget['used']+=1
        request_log={'attempt':request_budget['used'],'model':model,'upstream_http_status':None}
        requests.append(request_log)
        try:
            async with asyncio.timeout(PROVIDER_TIMEOUT_SECONDS):
                raw=await invoke(prompt,request_log)
        except TimeoutError as exc:
            request_log.update(request_duration_seconds=round(time.monotonic()-started,3),failure_category='timeout')
            logger.warning('LLM request model=%s duration_seconds=%.3f upstream_http_status=%s failure_category=%s',model,request_log['request_duration_seconds'],None,'timeout')
            raise httpx.ReadTimeout(f'Provider request exceeded {PROVIDER_TIMEOUT_SECONDS} seconds.') from exc
        except httpx.HTTPStatusError as exc:
            status=exc.response.status_code
            category='upstream_unavailable' if status==503 else 'upstream_http_error'
            request_log.update(request_duration_seconds=round(time.monotonic()-started,3),upstream_http_status=status,failure_category=category)
            logger.warning('LLM request model=%s duration_seconds=%.3f upstream_http_status=%s failure_category=%s',model,request_log['request_duration_seconds'],status,category)
            if status!=503 or os.getenv('LLM_PROVIDER','gemini')!='gemini' or request_budget['used']==MAX_PROVIDER_ATTEMPTS: raise
            delay=random.uniform(0,min(RETRY_BACKOFF_BASE_SECONDS*(2**retry_count),MAX_RETRY_BACKOFF_SECONDS))
            request_log['retry_delay_seconds']=round(delay,3)
            retry_count+=1
            await asyncio.sleep(delay)
        except httpx.TimeoutException as exc:
            request_log.update(request_duration_seconds=round(time.monotonic()-started,3),failure_category='timeout')
            logger.warning('LLM request model=%s duration_seconds=%.3f upstream_http_status=%s failure_category=%s',model,request_log['request_duration_seconds'],None,'timeout')
            raise
        except httpx.RequestError as exc:
            request_log.update(request_duration_seconds=round(time.monotonic()-started,3),failure_category='network_error')
            logger.warning('LLM request model=%s duration_seconds=%.3f upstream_http_status=%s failure_category=%s',model,request_log['request_duration_seconds'],None,'network_error')
            raise
        except HTTPException as exc:
            request_log.update(request_duration_seconds=round(time.monotonic()-started,3),failure_category='configuration_error')
            logger.warning('LLM request model=%s duration_seconds=%.3f upstream_http_status=%s failure_category=%s',model,request_log['request_duration_seconds'],None,'configuration_error')
            raise
        except (KeyError, IndexError, ValueError) as exc:
            request_log.update(request_duration_seconds=round(time.monotonic()-started,3),upstream_http_status=200,failure_category='invalid_response')
            logger.warning('LLM request model=%s duration_seconds=%.3f upstream_http_status=%s failure_category=%s',model,request_log['request_duration_seconds'],200,'invalid_response')
            raise
        else:
            request_log.update(request_duration_seconds=round(time.monotonic()-started,3),upstream_http_status=200,failure_category=None)
            logger.info('LLM request model=%s duration_seconds=%.3f upstream_http_status=%s failure_category=%s',model,request_log['request_duration_seconds'],200,'none')
            return raw
    raise HTTPException(503,'Gemini request attempt limit reached before a scene response was produced.')

@app.post('/api/scene')
async def create_scene(req:SceneRequest):
    if SLOTS.locked(): raise HTTPException(429,'A scene is being generated. Wait for it to finish.')
    async with SLOTS:
        start=time.monotonic();run_id=uuid.uuid4().hex
        record={'run_id':run_id,'script':req.script,'provider':os.getenv('LLM_PROVIDER','gemini'),'model':os.getenv('LLM_MODEL','').strip(),'attempts':[],'provider_requests':[]}
        request_budget={'used':0}
        prompt=req.script
        try:
            async with asyncio.timeout(SCENE_RESPONSE_DEADLINE_SECONDS):
                for attempt in range(2):
                    raw=await invoke_with_retries(prompt,record['provider_requests'],request_budget);record['attempts'].append({'raw':raw})
                    validation_started=time.monotonic()
                    try:
                        scene=Scene.model_validate_json(raw)
                    except ValidationError as exc:
                        validation_seconds=time.monotonic()-validation_started
                        record['attempts'][-1]['validation_seconds']=round(validation_seconds,3)
                        errors=exc.errors(include_url=False,include_context=False)
                        record['attempts'][-1]['errors']=errors
                        logger.warning('Scene stage=validation model=%s duration_seconds=%.3f failure_category=invalid_scene',record['model'],validation_seconds)
                        if attempt or request_budget['used']==MAX_PROVIDER_ATTEMPTS:
                            message='Model returned an invalid scene after repair.' if attempt else 'Model returned an invalid scene and the request attempt limit was reached before repair.'
                            raise HTTPException(422,{'message':message,'run_id':run_id,'errors':errors})
                        prompt=req.script+'\nRepair this draft:\n'+raw+'\nValidation errors:\n'+json.dumps(errors)
                        continue
                    validation_seconds=time.monotonic()-validation_started
                    warning_started=time.monotonic()
                    warnings=spatial_warnings(scene)
                    warning_seconds=time.monotonic()-warning_started
                    record['attempts'][-1].update(validation_seconds=round(validation_seconds,3),spatial_check_seconds=round(warning_seconds,3))
                    logger.info('Scene stage=validation model=%s duration_seconds=%.3f spatial_check_seconds=%.3f warnings=%d',record['model'],validation_seconds,warning_seconds,len(warnings))
                    if warnings and attempt==0:
                        record['attempts'][-1]['warnings']=warnings
                        if request_budget['used']<MAX_PROVIDER_ATTEMPTS:
                            prompt=req.script+'\nRepair this draft:\n'+raw+'\nIssues:\n'+json.dumps(warnings);continue
                    scene.description=req.script
                    record.update(status='success',scene=scene.model_dump(),warnings=warnings)
                    return {'run_id':run_id,'scene':scene.model_dump(),'warnings':warnings,'source':'live_model'}
        except TimeoutError as exc:
            message=f"Scene generation exceeded its {SCENE_RESPONSE_DEADLINE_SECONDS}-second deadline."
            record.update(status='provider_error',error={'failure_category':'scene_deadline_exceeded','message':message})
            raise HTTPException(504,message) from exc
        except HTTPException as exc:
            record.update(status='provider_error' if exc.status_code>=500 else 'failed')
            raise
        except httpx.HTTPStatusError as exc:
            response=exc.response
            try: provider_error=response.json().get('error',{})
            except (ValueError, AttributeError): provider_error={}
            if not isinstance(provider_error,dict): provider_error={}
            upstream_message=provider_error.get('message') or response.reason_phrase or 'No error message provided.'
            key=os.getenv('LLM_API_KEY','')
            if key: upstream_message=str(upstream_message).replace(key,'[REDACTED]')
            upstream_message=str(upstream_message)[:500]
            upstream_status=provider_error.get('status')
            detail=f"{record['provider']} API returned HTTP {response.status_code}"
            if upstream_status: detail+=f" ({upstream_status})"
            detail+=f" after {len(record['provider_requests'])} attempt(s): {upstream_message}"
            record.update(status='provider_error',error={'http_status':response.status_code,'provider_status':upstream_status,'failure_category':'upstream_unavailable' if response.status_code==503 else 'upstream_http_error','message':upstream_message})
            status_code=response.status_code if response.status_code in {400,401,403,404,429,503} else 502
            raise HTTPException(status_code,detail) from exc
        except httpx.TimeoutException as exc:
            message=f"{record['provider']} request timed out after {PROVIDER_TIMEOUT_SECONDS} seconds."
            record.update(status='provider_error',error={'type':type(exc).__name__,'failure_category':'timeout','message':message})
            raise HTTPException(504,message) from exc
        except httpx.RequestError as exc:
            message=f"{record['provider']} request failed due to a network error ({type(exc).__name__})."
            record.update(status='provider_error',error={'type':type(exc).__name__,'message':message})
            raise HTTPException(502,message) from exc
        except (KeyError, IndexError, ValueError) as exc:
            record['status']='provider_error'
            message=f"{record['provider']} returned an unexpected response ({type(exc).__name__})."
            record.update(error={'type':type(exc).__name__,'message':message})
            raise HTTPException(502,message) from exc
        finally:
            record['seconds']=round(time.monotonic()-start,3)
            logger.info('Scene stage=request_complete model=%s duration_seconds=%.3f provider_requests=%d outcome=%s',record['model'],record['seconds'],len(record['provider_requests']),record.get('status','failed'))
            DATA.mkdir(parents=True,exist_ok=True)
            (DATA/f'{run_id}.json').write_text(json.dumps(record,indent=2),encoding='utf-8')

# Build frontend first for one-server production use. Keep this local unless adding auth/rate limits.
DIST=BASE.parent/'frontend'/'dist'
if DIST.exists(): app.mount('/',StaticFiles(directory=DIST,html=True),name='frontend')
