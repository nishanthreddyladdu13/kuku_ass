from typing import Literal, Annotated
from pydantic import BaseModel, Field, ConfigDict, model_validator
import math

Number = Annotated[float, Field(ge=-50, le=50, allow_inf_nan=False)]
Vec = tuple[Number, Number, Number]
Size = tuple[Annotated[float, Field(gt=0, le=20)], Annotated[float, Field(gt=0, le=10)], Annotated[float, Field(gt=0, le=20)]]
Scale = tuple[Annotated[float, Field(gt=0, le=10)], Annotated[float, Field(gt=0, le=10)], Annotated[float, Field(gt=0, le=10)]]
class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
class Room(StrictModel):
    width: float = Field(ge=3, le=20)
    depth: float = Field(ge=3, le=20)
    height: float = Field(ge=2.2, le=6)
class Motion(StrictModel):
    end_position: Vec
    duration: float = Field(ge=1, le=10)
class Object(StrictModel):
    id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,50}$')
    name: str = Field(min_length=1, max_length=80)
    kind: Literal['character','table','chair','sofa','bed','counter','shelf','door','window','box']
    position: Vec
    dimensions: Size
    rotation: float = Field(ge=-360, le=360)
    color: str = Field(pattern=r'^#[0-9a-fA-F]{6}$')
    shape: Literal['rectangular','round'] = 'rectangular'
    pose: Literal['standing','seated'] = 'standing'
    motion: Motion | None = None
class Shot(StrictModel):
    id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,50}$')
    name: str = Field(min_length=1,max_length=80)
    start_time: float = Field(default=0,ge=0,le=120)
    duration: float = Field(default=5,gt=0,le=60)
    position: Vec
    target: Vec
    fov: float = Field(ge=20,le=95)
    roll: float = Field(default=0,ge=-180,le=180)
    action: str = Field(max_length=1500)
    @model_validator(mode='after')
    def distinct(self):
        if math.dist(self.position,self.target)<.05: raise ValueError('Camera and target must differ')
        return self
class Animation(StrictModel):
    id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,50}$')
    target_type: Literal['object','camera'] = 'object'
    object_id: str | None = Field(default=None,pattern=r'^[a-zA-Z0-9_-]{1,50}$')
    shot_id: str | None = Field(default=None,pattern=r'^[a-zA-Z0-9_-]{1,50}$')
    type: Literal['move','rotate','scale','appear','disappear','pan','tilt','dolly','track','zoom']
    start_time: float = Field(ge=0,le=120)
    duration: float = Field(gt=0,le=60)
    from_position: Vec | None = None
    to_position: Vec | None = None
    from_rotation: float | None = Field(default=None,ge=-360,le=360)
    to_rotation: float | None = Field(default=None,ge=-360,le=360)
    from_scale: Scale | None = None
    to_scale: Scale | None = None
    from_target: Vec | None = None
    to_target: Vec | None = None
    from_fov: float | None = Field(default=None,ge=20,le=95)
    to_fov: float | None = Field(default=None,ge=20,le=95)
    target_id: str | None = Field(default=None,pattern=r'^[a-zA-Z0-9_-]{1,50}$')
    @model_validator(mode='after')
    def fields_match_type(self):
        required={
            'move':('from_position','to_position'),
            'rotate':('from_rotation','to_rotation'),
            'scale':('from_scale','to_scale'),
            'pan':('from_target','to_target'),
            'tilt':('from_target','to_target'),
            'dolly':('from_position','to_position'),
            'track':('from_position','to_position','from_target','to_target'),
            'zoom':('from_fov','to_fov'),
        }.get(self.type,())
        if any(getattr(self,name) is None for name in required):
            raise ValueError(f'{self.type} animation requires {", ".join(required)}')
        return self
class Relation(StrictModel):
    subject: str
    relation: Literal['left_of','right_of','near','faces']
    target: str
class Scene(StrictModel):
    version: Literal[2] = 2
    title: str = Field(min_length=1,max_length=100)
    description: str = Field(max_length=8000)
    room: Room
    objects: list[Object] = Field(min_length=1,max_length=20)
    shots: list[Shot] = Field(min_length=1,max_length=4)
    assumptions: list[str] = Field(default_factory=list,max_length=20)
    unsupported: list[str] = Field(default_factory=list,max_length=20)
    relations: list[Relation] = Field(default_factory=list,max_length=30)
    animations: list[Animation] = Field(default_factory=list,max_length=100)
    @model_validator(mode='after')
    def consistent(self):
        for values in [self.objects,self.shots,self.animations]:
            ids=[v.id for v in values]
            if len(ids)!=len(set(ids)): raise ValueError('Duplicate IDs')
        if sum(o.kind=='character' for o in self.objects)>4: raise ValueError('At most four principal characters')
        objects={o.id:o for o in self.objects}
        ids=set(objects)
        shot_ids={s.id for s in self.shots}
        shots={s.id:s for s in self.shots}
        tracks={}
        for animation in self.animations:
            if animation.target_type=='object':
                if animation.object_id not in ids or animation.shot_id is not None: raise ValueError('Object animations must reference one existing object')
                if animation.type not in {'move','rotate','scale','appear','disappear'}: raise ValueError('Unsupported object animation type')
                obj=objects[animation.object_id]
                if obj.motion and animation.type=='move': raise ValueError('Use either object motion or move animations, not both')
                if animation.type=='move': channels={'position':(animation.from_position,animation.to_position)}
                elif animation.type=='rotate': channels={'rotation':(animation.from_rotation,animation.to_rotation)}
                elif animation.type=='scale': channels={'scale':(animation.from_scale,animation.to_scale)}
                else: channels={'visibility':((0,1) if animation.type=='appear' else (1,0))}
                initial={'position':obj.position,'rotation':obj.rotation,'scale':(1,1,1),'visibility':1}
            else:
                if animation.shot_id not in shot_ids or animation.object_id is not None: raise ValueError('Camera animations must reference one existing shot')
                if animation.type not in {'pan','tilt','dolly','track','zoom'}: raise ValueError('Unsupported camera animation type')
                shot=shots[animation.shot_id]
                if animation.type in {'pan','tilt'}: channels={'target':(animation.from_target,animation.to_target)}
                elif animation.type=='dolly': channels={'position':(animation.from_position,animation.to_position)}
                elif animation.type=='track': channels={'position':(animation.from_position,animation.to_position),'target':(animation.from_target,animation.to_target)}
                else: channels={'fov':(animation.from_fov,animation.to_fov)}
                initial={'position':shot.position,'target':shot.target,'fov':shot.fov}
            for channel,values in channels.items():
                key=(animation.target_type,animation.object_id or animation.shot_id,channel)
                tracks.setdefault(key,[]).append((animation,values[0],values[1],initial[channel]))
            if animation.target_id is not None and animation.target_id not in ids: raise ValueError('Animation references missing target object')
        for entries in tracks.values():
            entries.sort(key=lambda entry:entry[0].start_time)
            previous_end=None
            previous_value=None
            for animation,start_value,end_value,initial_value in entries:
                if previous_end is not None:
                    if animation.start_time<previous_end: raise ValueError('Animations affecting the same transform cannot overlap')
                    expected=previous_value
                else: expected=initial_value
                if isinstance(start_value,(tuple,list)):
                    continuous=len(start_value)==len(expected) and all(math.isclose(a,b,abs_tol=.01) for a,b in zip(start_value,expected))
                else: continuous=math.isclose(start_value,expected,abs_tol=.01)
                if not continuous: raise ValueError('Animation start state must match the preceding transform state')
                previous_end=animation.start_time+animation.duration
                previous_value=end_value
        for r in self.relations:
            if r.subject not in ids or r.target not in ids: raise ValueError('Relation references missing object')
        return self
class SceneRequest(StrictModel):
    script: str = Field(min_length=10,max_length=8000)
class SceneImageRequest(StrictModel):
    scene: Scene
    shot_id: str = Field(pattern=r'^[a-zA-Z0-9_-]{1,50}$')

def spatial_warnings(scene: Scene):
    warnings=[]
    for o in scene.objects:
        w,h,d=o.dimensions; a=math.radians(o.rotation)
        hx=(abs(w*math.cos(a))+abs(d*math.sin(a)))/2
        hz=(abs(w*math.sin(a))+abs(d*math.cos(a)))/2
        for pos in [o.position]+([o.motion.end_position] if o.motion else []):
            x,y,z=pos
            if abs(x)+hx>scene.room.width/2+.15 or abs(z)+hz>scene.room.depth/2+.15 or y<0 or y+h>scene.room.height+.15:
                warnings.append(f'{o.name}: geometry or movement endpoint extends outside the room.')
    for animation in scene.animations:
        positions=[]
        if animation.target_type=='object' and animation.type=='move':
            positions=[animation.from_position,animation.to_position]
            obj=next(o for o in scene.objects if o.id==animation.object_id)
            w,h,d=obj.dimensions; angle=math.radians(obj.rotation)
            hx=(abs(w*math.cos(angle))+abs(d*math.sin(angle)))/2
            hz=(abs(w*math.sin(angle))+abs(d*math.cos(angle)))/2
            if any(abs(p[0])+hx>scene.room.width/2+.15 or abs(p[2])+hz>scene.room.depth/2+.15 or p[1]<0 or p[1]+h>scene.room.height+.15 for p in positions):
                warnings.append(f'{obj.name}: animation endpoint extends outside the room.')
        elif animation.target_type=='camera' and animation.type in {'dolly','track'}:
            positions=[animation.from_position,animation.to_position]
            if any(abs(p[0])>=scene.room.width/2 or abs(p[2])>=scene.room.depth/2 or p[1]<0 or p[1]>scene.room.height for p in positions):
                warnings.append(f'{animation.shot_id}: camera animation leaves the room.')
    objects={o.id:o for o in scene.objects}
    for r in scene.relations:
        a,b=objects[r.subject],objects[r.target]
        if r.relation=='left_of' and a.position[0]>=b.position[0]: warnings.append(f'{a.name} is not left of {b.name} in world coordinates.')
        if r.relation=='right_of' and a.position[0]<=b.position[0]: warnings.append(f'{a.name} is not right of {b.name} in world coordinates.')
        if r.relation=='near' and math.dist(a.position,b.position)>2: warnings.append(f'{a.name} is more than 2m from {b.name}.')
        if r.relation=='faces':
            dx,dz=b.position[0]-a.position[0],b.position[2]-a.position[2]
            length=math.hypot(dx,dz)
            angle=math.radians(a.rotation)
            if length and (math.sin(angle)*dx+math.cos(angle)*dz)/length < .7: warnings.append(f'{a.name} does not face {b.name}.')
    for s in scene.shots:
        if abs(s.position[0])>=scene.room.width/2 or abs(s.position[2])>=scene.room.depth/2:
            warnings.append(f'{s.name}: camera is outside the room; a wall may block the view.')
    return list(dict.fromkeys(warnings))
