import * as T from 'three';
import {type Project,type Shot,WIDTH,HEIGHT,NEAR,FAR} from './model';
export function cameraFor(s:Shot){const c=new T.PerspectiveCamera(s.fov,WIDTH/HEIGHT,NEAR,FAR);c.position.fromArray(s.position);c.lookAt(new T.Vector3(...s.target));c.rotateZ(T.MathUtils.degToRad(s.roll));c.updateMatrixWorld(true);return c;}
export function buildScene(p:Project){
 const scene=new T.Scene();scene.background=new T.Color('#b7c0b7');
 scene.add(new T.HemisphereLight('#fff3d9','#596b68',2.5));const light=new T.DirectionalLight('#ffedce',4);light.position.set(-1,5,3);scene.add(light);
 const mat=(color:string)=>new T.MeshStandardMaterial({color,roughness:.85});
 const box=(parent:T.Object3D,w:number,h:number,d:number,x:number,y:number,z:number,color:string)=>{const m=new T.Mesh(new T.BoxGeometry(w,h,d),mat(color));m.position.set(x,y,z);parent.add(m);return m;};
 const {width:w,depth:d,height:h}=p.room;
 box(scene,w,.12,d,0,-.06,0,'#a39886');
 // Open ceiling; outward faces remain solid. Use inside cameras for exports.
 box(scene,w,h,.08,0,h/2,-d/2,'#c7c9bf');box(scene,w,h,.08,0,h/2,d/2,'#c7c9bf');
 box(scene,.08,h,d,-w/2,h/2,0,'#a3b3b2');box(scene,.08,h,d,w/2,h/2,0,'#a3b3b2');
 for(const o of p.objects){const g=new T.Group();g.name=o.id;g.position.fromArray(o.position);g.rotation.y=T.MathUtils.degToRad(o.rotation);scene.add(g);const [a,b,c]=o.dimensions;const draw=(ww:number,hh:number,dd:number,x:number,y:number,z:number,color=o.color)=>box(g,ww,hh,dd,x,y,z,color);
  if(o.kind==='character'){
   // Normalized articulated proxy; dimensions and pose remain editable.
   const seated=o.pose==='seated',base=seated?1.4:1.8;
   const head=new T.Mesh(new T.SphereGeometry(.17,12,8),mat('#d4ad8a'));head.position.set(0,base-.18,0);g.add(head);
   draw(.4,.5,.25,0,base-.62,0);draw(.11,.5,.11,-.28,base-.65,.03);draw(.11,.5,.11,.28,base-.65,.03);
   for(const x of[-.12,.12]){if(seated){draw(.15,.15,.45,x,.53,.17);draw(.14,.48,.14,x,.25,.34);}else draw(.16,.75,.18,x,.38,0,'#475563');}
   g.scale.set(a/.6,b/base,c/.6);
  }else if(o.kind==='table'){
   if(o.shape==='round'){const m=new T.Mesh(new T.CylinderGeometry(.5,.5,.1,32),mat(o.color));m.scale.set(a,1,c);m.position.y=b-.05;g.add(m);draw(.15,b-.1,.15,0,(b-.1)/2,0);}else{draw(a,.1,c,0,b-.05,0);for(const x of[-a/2+.08,a/2-.08])for(const z of[-c/2+.08,c/2-.08])draw(.08,b-.1,.08,x,(b-.1)/2,z);}
  }else if(o.kind==='chair'||o.kind==='sofa'){
   draw(a,b*.2,c,0,b*.45,0);draw(a,b*.5,c*.18,0,b*.75,-c*.41);for(const x of[-a*.4,a*.4])for(const z of[-c*.35,c*.35])draw(.06,b*.35,.06,x,b*.175,z);
  }else if(o.kind==='bed'){draw(a,b*.7,c,0,b*.35,0);draw(a,b*.3,c,0,b*.85,0,'#e9e4d6');draw(a*.65,.1,c*.2,0,b+.05,-c*.3,'#faf5e9');
  }else if(o.kind==='shelf'){for(const x of[-a/2+.04,a/2-.04])draw(.08,b,c,x,b/2,0);for(let i=0;i<4;i++)draw(a,.07,c,0,.06+i*(b-.12)/3,0);
  }else if(o.kind==='window'){draw(a,b,c,0,b/2,0,'#88b9cb');draw(.05,b,c+.02,0,b/2,0,'#f0e7d5');draw(a,.05,c+.02,0,b/2,0,'#f0e7d5');
  }else{draw(a,b,c,0,b/2,0);}
 }
 return scene;
}
export function disposeScene(s:T.Scene){s.traverse(o=>{if(o instanceof T.Mesh){o.geometry.dispose();(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>m.dispose());}});}
export function metadata(p:Project,s:Shot){const c=cameraFor(s);return {scene:p,shot:s,dimensions:{width:WIDTH,height:HEIGHT},coordinates:{units:'metres',up:'+Y',origin:'room centre',position:'object bottom centre',rotation:'degrees around Y, 0 faces +Z'},camera:{matrixWorld:c.matrixWorld.toArray(),projectionMatrix:c.projectionMatrix.toArray(),near:NEAR,far:FAR},depth:{encoding:'8-bit linear camera-forward distance, white near / black far',formula:'255 * (1 - clamp((distance - near)/(far - near),0,1))',modelCompatibility:'Not tested; not an input to the selected manual image-to-video route.'},generation:{status:'manual_external',videoModelTested:false}};}
