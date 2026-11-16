import {test} from 'node:test';import assert from 'node:assert/strict';import fs from 'node:fs';import * as T from 'three';
import {projectSchema,atTime,objectAnimationAt,cameraAtTime,sceneDuration,depthByte,type V3} from '../src/model';import {applySceneTime,buildScene,cameraFor,disposeScene} from '../src/scene';import {raster} from '../src/raster';
const read=(n:number)=>projectSchema.parse(JSON.parse(fs.readFileSync(`../backend/examples/${n}.json`,'utf8')));
test('Different scenes preserve schema through serialization',()=>{for(const n of[1,2,3]){const p=read(n);assert.deepEqual(projectSchema.parse(JSON.parse(JSON.stringify(p))),p);}});
test('Movement changes only moving object and leaves source unchanged',()=>{const p=read(2),before=JSON.stringify(p),a=atTime(p,4);assert.notDeepEqual(a.objects[2].position,p.objects[2].position);assert.deepEqual(a.objects[0],p.objects[0]);assert.equal(JSON.stringify(p),before);});
test('Round and rectangular table use different geometry',()=>{const p=read(1),s=buildScene(p);assert.ok(s.getObjectByName('round-table')!.children.some(o=>o instanceof T.Mesh&&o.geometry.type==='CylinderGeometry'));disposeScene(s);});
test('Five sampled rendered depths agree with independent ray intersections',()=>{const p=read(1),s=buildScene(p);s.updateMatrixWorld(true);for(const shot of p.shots){const c=cameraFor(shot),pix=raster(s,c,144,256,true);for(const[x,y]of[[72,128],[36,64],[108,192],[72,210],[95,100]]){const ray=new T.Raycaster();ray.setFromCamera(new T.Vector2(2*(x+.5)/144-1,1-2*(y+.5)/256),c);const hit=ray.intersectObject(s,true)[0];assert.ok(hit);const expected=depthByte(-hit.point.clone().applyMatrix4(c.matrixWorldInverse).z);assert.ok(Math.abs(expected-pix[(y*144+x)*4])<=1);}}disposeScene(s);});
test('All three sets render different previews',()=>{const values=[1,2,3].map(n=>{const p=read(n),s=buildScene(p),img=raster(s,cameraFor(p.shots[0]),72,128,false);disposeScene(s);return Buffer.from(img).toString('base64');});assert.equal(new Set(values).size,3);});
test('Animation schema accepts shot timing and deterministic object and camera tracks',()=>{
 const p=read(2),object=p.objects.find(candidate=>!candidate.motion)!,shot=p.shots[0],start:V3=[...object.position],end:V3=[start[0]+.5,start[1],start[2]];
 p.shots[0]={...shot,start_time:1,duration:3};
 p.animations=[
  {id:'table-move',target_type:'object',object_id:object.id,shot_id:null,type:'move',start_time:1,duration:2,from_position:start,to_position:end,from_rotation:null,to_rotation:null,from_scale:null,to_scale:null,from_target:null,to_target:null,from_fov:null,to_fov:null,target_id:null},
  {id:'camera-zoom',target_type:'camera',object_id:null,shot_id:shot.id,type:'zoom',start_time:1,duration:2,from_position:null,to_position:null,from_rotation:null,to_rotation:null,from_scale:null,to_scale:null,from_target:null,to_target:null,from_fov:shot.fov,to_fov:shot.fov+6,target_id:null},
 ];
 const scene=projectSchema.parse(p),animatedObject=scene.objects.find(candidate=>candidate.id===object.id)!,objectState=objectAnimationAt(scene,animatedObject,2),cameraState=cameraAtTime(scene,scene.shots[0],2);
 assert.deepEqual(objectState.position,[start[0]+.25,start[1],start[2]]);
 assert.equal(cameraState.fov,shot.fov+3);
 assert.equal(sceneDuration(scene),5);
 const rendered=buildScene(scene);applySceneTime(rendered,scene,2);
 assert.equal(rendered.getObjectByName(object.id)!.position.x,start[0]+.25);
 disposeScene(rendered);
});
