import {z} from 'zod';

export const vec=z.tuple([
 z.number().finite().min(-50).max(50),
 z.number().finite().min(-50).max(50),
 z.number().finite().min(-50).max(50),
]);
const scaleVec=z.tuple([
 z.number().finite().positive().max(10),
 z.number().finite().positive().max(10),
 z.number().finite().positive().max(10),
]);
export const objectSchema=z.object({
 id:z.string().regex(/^[a-zA-Z0-9_-]{1,50}$/),
 name:z.string().min(1).max(80),
 kind:z.enum(['character','table','chair','sofa','bed','counter','shelf','door','window','box']),
 position:vec,
 dimensions:z.tuple([z.number().positive().max(20),z.number().positive().max(10),z.number().positive().max(20)]),
 rotation:z.number().finite().min(-360).max(360),
 color:z.string().regex(/^#[0-9a-fA-F]{6}$/),
 shape:z.enum(['rectangular','round']).default('rectangular'),
 pose:z.enum(['standing','seated']).default('standing'),
 motion:z.object({end_position:vec,duration:z.number().min(1).max(10)}).nullable().default(null),
}).strict();
export const shotSchema=z.object({
 id:z.string().regex(/^[a-zA-Z0-9_-]{1,50}$/),
 name:z.string().min(1).max(80),
 start_time:z.number().finite().min(0).max(120).default(0),
 duration:z.number().finite().positive().max(60).default(5),
 position:vec,
 target:vec,
 fov:z.number().min(20).max(95),
 roll:z.number().min(-180).max(180).default(0),
 action:z.string().max(1500),
}).strict().refine(s=>Math.hypot(...s.position.map((v,i)=>v-s.target[i]))>.05,'Camera and target must differ');
const objectAnimationTypes=['move','rotate','scale','appear','disappear'] as const;
const cameraAnimationTypes=['pan','tilt','dolly','track','zoom'] as const;
export const animationSchema=z.object({
 id:z.string().regex(/^[a-zA-Z0-9_-]{1,50}$/),
 target_type:z.enum(['object','camera']).default('object'),
 object_id:z.string().nullable().optional().default(null),
 shot_id:z.string().nullable().optional().default(null),
 type:z.enum([...objectAnimationTypes,...cameraAnimationTypes]),
 start_time:z.number().finite().min(0).max(120),
 duration:z.number().finite().positive().max(60),
 from_position:vec.nullable().optional().default(null),
 to_position:vec.nullable().optional().default(null),
 from_rotation:z.number().finite().min(-360).max(360).nullable().optional().default(null),
 to_rotation:z.number().finite().min(-360).max(360).nullable().optional().default(null),
 from_scale:scaleVec.nullable().optional().default(null),
 to_scale:scaleVec.nullable().optional().default(null),
 from_target:vec.nullable().optional().default(null),
 to_target:vec.nullable().optional().default(null),
 from_fov:z.number().finite().min(20).max(95).nullable().optional().default(null),
 to_fov:z.number().finite().min(20).max(95).nullable().optional().default(null),
 target_id:z.string().regex(/^[a-zA-Z0-9_-]{1,50}$/).nullable().optional().default(null),
}).strict().superRefine((animation,context)=>{
 const required:Partial<Record<typeof animation.type,(keyof typeof animation)[]>>={
  move:['from_position','to_position'],
  rotate:['from_rotation','to_rotation'],
  scale:['from_scale','to_scale'],
  pan:['from_target','to_target'],
  tilt:['from_target','to_target'],
  dolly:['from_position','to_position'],
  track:['from_position','to_position','from_target','to_target'],
  zoom:['from_fov','to_fov'],
 };
 for(const key of required[animation.type]||[])if(animation[key]===null)context.addIssue({code:'custom',path:[key],message:`${animation.type} requires ${key}`});
 if(animation.target_type==='object'){
  if(!animation.object_id||animation.shot_id!==null)context.addIssue({code:'custom',path:['object_id'],message:'Object animations require object_id and no shot_id'});
  if(!(objectAnimationTypes as readonly string[]).includes(animation.type))context.addIssue({code:'custom',path:['type'],message:'Unsupported object animation type'});
 }else{
  if(!animation.shot_id||animation.object_id!==null)context.addIssue({code:'custom',path:['shot_id'],message:'Camera animations require shot_id and no object_id'});
  if(!(cameraAnimationTypes as readonly string[]).includes(animation.type))context.addIssue({code:'custom',path:['type'],message:'Unsupported camera animation type'});
 }
});
export const projectSchema=z.object({
 version:z.literal(2),
 title:z.string().min(1).max(100),
 description:z.string().max(8000),
 room:z.object({width:z.number().min(3).max(20),depth:z.number().min(3).max(20),height:z.number().min(2.2).max(6)}).strict(),
 objects:z.array(objectSchema).min(1).max(20),
 shots:z.array(shotSchema).min(1).max(4),
 assumptions:z.array(z.string()).max(20).default([]),
 unsupported:z.array(z.string()).max(20).default([]),
 relations:z.array(z.object({subject:z.string(),relation:z.enum(['left_of','right_of','near','faces']),target:z.string()}).strict()).max(30).default([]),
 animations:z.array(animationSchema).max(100).default([]),
}).strict().superRefine((project,context)=>{
 for(const values of[project.objects,project.shots,project.animations]){
  if(new Set(values.map(value=>value.id)).size!==values.length)context.addIssue({code:'custom',message:'Duplicate IDs'});
 }
 if(project.objects.filter(o=>o.kind==='character').length>4)context.addIssue({code:'custom',message:'At most four characters'});
 for(const relation of project.relations)if(![relation.subject,relation.target].every(id=>project.objects.some(o=>o.id===id)))context.addIssue({code:'custom',message:'Missing relation target'});
 const objects=new Set(project.objects.map(object=>object.id));
 const shots=new Set(project.shots.map(shot=>shot.id));
 for(const [index,animation] of project.animations.entries()){
  if(animation.target_type==='object'&&animation.object_id&&!objects.has(animation.object_id))context.addIssue({code:'custom',path:['animations',index,'object_id'],message:'Animation references a missing object'});
  if(animation.target_type==='camera'&&animation.shot_id&&!shots.has(animation.shot_id))context.addIssue({code:'custom',path:['animations',index,'shot_id'],message:'Animation references a missing shot'});
  if(animation.target_id&&!objects.has(animation.target_id))context.addIssue({code:'custom',path:['animations',index,'target_id'],message:'Animation references a missing target'});
 }
});
export type Project=z.infer<typeof projectSchema>;
export type Shot=z.infer<typeof shotSchema>;
export type SceneObject=z.infer<typeof objectSchema>;
export type Animation=z.infer<typeof animationSchema>;
export type V3=[number,number,number];
export const WIDTH=576,HEIGHT=1024,NEAR=.1,FAR=40;
export function depthByte(z:number){return Math.round(255*(1-Math.min(1,Math.max(0,(z-NEAR)/(FAR-NEAR)))));}

export function objectAnimationAt(project:Project,object:SceneObject,time:number){
 let position:[number,number,number]=[...object.position],rotation=object.rotation,scale:V3=[1,1,1],opacity=1;
 if(object.motion){const t=Math.max(0,Math.min(1,time/object.motion.duration));position=position.map((value,index)=>value+(object.motion!.end_position[index]-value)*t)as V3;}
 const tracks=project.animations.filter(animation=>animation.target_type==='object'&&animation.object_id===object.id).sort((a,b)=>a.start_time-b.start_time);
 for(const animation of tracks){
  if(animation.type==='move'&&animation.from_position&&animation.to_position&&time>=animation.start_time){const t=Math.max(0,Math.min(1,(time-animation.start_time)/animation.duration));position=animation.from_position.map((value,index)=>value+(animation.to_position![index]-value)*t)as V3;}
  if(animation.type==='rotate'&&animation.from_rotation!==null&&animation.to_rotation!==null&&time>=animation.start_time){const t=Math.max(0,Math.min(1,(time-animation.start_time)/animation.duration));rotation=animation.from_rotation+(animation.to_rotation-animation.from_rotation)*t;}
  if(animation.type==='scale'&&animation.from_scale&&animation.to_scale&&time>=animation.start_time){const t=Math.max(0,Math.min(1,(time-animation.start_time)/animation.duration));scale=animation.from_scale.map((value,index)=>value+(animation.to_scale![index]-value)*t)as V3;}
  if(animation.type==='appear')opacity=time<animation.start_time?0:Math.min(1,(time-animation.start_time)/animation.duration);
  if(animation.type==='disappear')opacity=time<animation.start_time?1:1-Math.min(1,(time-animation.start_time)/animation.duration);
 }
 return{position,rotation,scale,opacity};
}
export function cameraAtTime(project:Project,shot:Shot,time:number){
 let position:V3=[...shot.position],target:V3=[...shot.target],fov=shot.fov;
 const tracks=project.animations.filter(animation=>animation.target_type==='camera'&&animation.shot_id===shot.id).sort((a,b)=>a.start_time-b.start_time);
 for(const animation of tracks){
  const t=Math.max(0,Math.min(1,(time-animation.start_time)/animation.duration));
  if(time<animation.start_time)continue;
  if((animation.type==='pan'||animation.type==='tilt')&&animation.from_target&&animation.to_target)target=animation.from_target.map((value,index)=>value+(animation.to_target![index]-value)*t)as V3;
  if((animation.type==='dolly'||animation.type==='track')&&animation.from_position&&animation.to_position)position=animation.from_position.map((value,index)=>value+(animation.to_position![index]-value)*t)as V3;
  if((animation.type==='track')&&animation.from_target&&animation.to_target)target=animation.from_target.map((value,index)=>value+(animation.to_target![index]-value)*t)as V3;
  if(animation.type==='zoom'&&animation.from_fov!==null&&animation.to_fov!==null)fov=animation.from_fov+(animation.to_fov-animation.from_fov)*t;
 }
 return{...shot,position,target,fov};
}
export function sceneDuration(project:Project){
 return Math.max(5,...project.shots.map(shot=>shot.start_time+shot.duration),...project.objects.map(object=>object.motion?.duration||0),...project.animations.map(animation=>animation.start_time+animation.duration));
}
export function atTime(project:Project,time:number):Project{
 return{...project,objects:project.objects.map(object=>{
  const state=objectAnimationAt(project,object,time);
  return{...object,position:state.position,rotation:state.rotation};
 })};
}
