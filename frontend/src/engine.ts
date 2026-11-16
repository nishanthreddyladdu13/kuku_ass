import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {applySceneTime,buildScene,cameraFor,disposeScene,metadata} from './scene';
import {raster} from './raster';
import {cameraAtTime,type Project,type Shot,sceneDuration,WIDTH,HEIGHT,NEAR,FAR,depthByte} from './model';

export class Studio{
 scene:T.Scene=new T.Scene();
 camera=cameraFor({id:'init',name:'Init',start_time:0,duration:5,position:[.68,1.55,2.15],target:[0,1.15,-1.05],fov:51,roll:0,action:''});
 freeCamera=new T.PerspectiveCamera(50,1,.1,100);
 controls:OrbitControls;
 renderer:T.WebGLRenderer|null=null;
 canvas:HTMLCanvasElement;
 free=false;
 depth=false;
 private frame=0;
 private observer:ResizeObserver;
 private lastKey='';
 private revision=0;
 private project:Project|null=null;
 private selectedShot:Shot|null=null;
 private time=0;
 depthMaterial:T.ShaderMaterial;
 constructor(public host:HTMLElement){
  try{
   this.renderer=new T.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
   this.renderer.outputColorSpace=T.SRGBColorSpace;
   this.renderer.toneMapping=T.ACESFilmicToneMapping;
   this.renderer.toneMappingExposure=1.25;
   this.canvas=this.renderer.domElement;
  }catch{this.canvas=document.createElement('canvas');}
  this.canvas.setAttribute('aria-label','Interactive 3D scene viewport');
  host.appendChild(this.canvas);
  this.freeCamera.position.set(8,12,8);
  this.controls=new OrbitControls(this.freeCamera,this.canvas);
  this.controls.target.set(0,1,0);
  this.controls.enableDamping=true;
  this.controls.enabled=false;
  this.depthMaterial=new T.ShaderMaterial({
   uniforms:{nearPlane:{value:NEAR},farPlane:{value:FAR}},
   vertexShader:'varying float eyeDepth;void main(){vec4 v=modelViewMatrix*vec4(position,1.0);eyeDepth=-v.z;gl_Position=projectionMatrix*v;}',
   fragmentShader:'uniform float nearPlane;uniform float farPlane;varying float eyeDepth;void main(){float d=1.0-clamp((eyeDepth-nearPlane)/(farPlane-nearPlane),0.0,1.0);gl_FragColor=vec4(vec3(d),1.0);}',
   toneMapped:false,
  });
  this.observer=new ResizeObserver(()=>this.fit());
  this.observer.observe(host);
  this.fit();
  this.tick();
 }
 fit(){
  const w=this.host.clientWidth,h=this.host.clientHeight;
  this.canvas.width=w;this.canvas.height=h;
  this.renderer?.setSize(w,h);
  this.freeCamera.aspect=w/h;
  this.freeCamera.updateProjectionMatrix();
  this.lastKey='';
 }
 load(project:Project){
  disposeScene(this.scene);
  this.project=project;
  this.selectedShot=project.shots[0];
  this.scene=buildScene(project);
  this.revision++;
  this.setTime(this.time);
 }
 setShot(shot:Shot){
  this.selectedShot=shot;
  this.setTime(this.time);
 }
 setTime(time:number){
  this.time=Math.max(0,Math.min(this.project?sceneDuration(this.project):0,time));
  if(this.project)applySceneTime(this.scene,this.project,this.time);
  if(this.project&&this.selectedShot)this.camera=cameraFor(cameraAtTime(this.project,this.selectedShot,this.time));
  this.lastKey='';
 }
 private tick=()=>{
  this.frame=requestAnimationFrame(this.tick);
  this.controls.enabled=this.free;
  this.controls.update();
  this.freeCamera.updateMatrixWorld();
  const camera=this.free?this.freeCamera:this.camera;
  const width=this.host.clientWidth,height=this.host.clientHeight;
  const frameHeight=this.free?height:Math.round(Math.min(height,width*HEIGHT/WIDTH));
  const frameWidth=this.free?width:Math.round(frameHeight*WIDTH/HEIGHT);
  const x=Math.floor((width-frameWidth)/2),y=Math.floor((height-frameHeight)/2);
  const key=[this.revision,this.free,this.depth,this.time,frameWidth,frameHeight,camera.matrixWorld.elements.join(','),camera.projectionMatrix.elements.join(',')].join('|');
  if(key===this.lastKey)return;
  this.lastKey=key;
  if(this.renderer){
   const renderer=this.renderer,background=this.scene.background;
   renderer.setRenderTarget(null);renderer.setScissorTest(false);
   renderer.setViewport(0,0,width,height);renderer.setClearColor('#171e1e');renderer.clear();
   renderer.setViewport(x,y,frameWidth,frameHeight);renderer.setScissor(x,y,frameWidth,frameHeight);renderer.setScissorTest(true);
   this.scene.overrideMaterial=this.depth?this.depthMaterial:null;
   if(this.depth)this.scene.background=new T.Color(0);
   renderer.render(this.scene,camera);
   this.scene.overrideMaterial=null;this.scene.background=background;renderer.setScissorTest(false);
  }else{
   const context=this.canvas.getContext('2d')!;
   context.fillStyle='#171e1e';context.fillRect(0,0,width,height);
   context.putImageData(new ImageData(new Uint8ClampedArray(raster(this.scene,camera,frameWidth,frameHeight,this.depth)),frameWidth,frameHeight),x,y);
  }
 };
 pixels(shot:Shot,depth:boolean,width=WIDTH,height=HEIGHT){
  const camera=this.project?cameraFor(cameraAtTime(this.project,shot,this.time)):cameraFor(shot);
  if(!this.renderer)return raster(this.scene,camera,width,height,depth);
  const renderer=this.renderer;
  const target=new T.WebGLRenderTarget(width,height,{depthBuffer:true,format:T.RGBAFormat,type:T.UnsignedByteType});
  target.texture.colorSpace=depth?T.NoColorSpace:T.SRGBColorSpace;
  const background=this.scene.background;
  this.scene.overrideMaterial=depth?this.depthMaterial:null;
  if(depth)this.scene.background=new T.Color(0);
  renderer.setScissorTest(false);renderer.setRenderTarget(target);renderer.setViewport(0,0,width,height);renderer.clear();
  renderer.render(this.scene,camera);
  const raw=new Uint8Array(width*height*4);
  renderer.readRenderTargetPixels(target,0,0,width,height,raw);
  renderer.setRenderTarget(null);this.scene.overrideMaterial=null;this.scene.background=background;target.dispose();this.lastKey='';
  const output=new Uint8Array(raw.length);
  for(let row=0;row<height;row++)output.set(raw.subarray((height-1-row)*width*4,(height-row)*width*4),row*width*4);
  return output;
 }
 png(shot:Shot,depth:boolean,width=WIDTH,height=HEIGHT){
  const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
  canvas.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(this.pixels(shot,depth,width,height)),width,height),0,0);
  return canvas.toDataURL('image/png');
 }
 check(shot:Shot){
  const width=144,height=256,pixels=this.pixels(shot,true,width,height),camera=this.project?cameraFor(cameraAtTime(this.project,shot,this.time)):cameraFor(shot),raycaster=new T.Raycaster();
  this.scene.updateMatrixWorld(true);
  const samples=[];
  for(const[x,y]of[[72,128],[36,64],[108,192],[72,210],[95,100]]){
   raycaster.setFromCamera(new T.Vector2(2*(x+.5)/width-1,1-2*(y+.5)/height),camera);
   const hit=raycaster.intersectObject(this.scene,true)[0];
   if(hit){
    const z=-hit.point.clone().applyMatrix4(camera.matrixWorldInverse).z,expected=depthByte(z),actual=pixels[(y*width+x)*4];
    samples.push({x,y,expected,actual,error:Math.abs(expected-actual)});
   }
  }
  return{passed:samples.length===5&&samples.every(sample=>sample.error<=1),samples};
 }
 dispose(){
  cancelAnimationFrame(this.frame);this.observer.disconnect();this.controls.dispose();
  disposeScene(this.scene);this.depthMaterial.dispose();this.renderer?.dispose();this.canvas.remove();
 }
}
export function bytes(url:string){return Uint8Array.from(atob(url.split(',')[1]),character=>character.charCodeAt(0));}
export{metadata};
