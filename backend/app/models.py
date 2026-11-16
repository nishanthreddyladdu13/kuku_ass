from typing import Literal, Annotated
from pydantic import BaseModel, Field, ConfigDict, model_validator
import math

Number = Annotated[float, Field(ge=-50, le=50, allow_inf_nan=False)]
Vec = tuple[Number, Number, Number]
Size = tuple[Annotated[float, Field(gt=0, le=20)], Annotated[float, Field(gt=0, le=10)], Annotated[float, Field(gt=0, le=20)]]
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
    position: Vec
    target: Vec
    fov: float = Field(ge=20,le=95)
    roll: float = Field(default=0,ge=-180,le=180)
    action: str = Field(max_length=1500)
    @model_validator(mode='after')
    def distinct(self):
        if math.dist(self.position,self.target)<.05: raise ValueError('Camera and target must differ')
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
    @model_validator(mode='after')
    def consistent(self):
        for values in [self.objects,self.shots]:
            ids=[v.id for v in values]
            if len(ids)!=len(set(ids)): raise ValueError('Duplicate IDs')
        if sum(o.kind=='character' for o in self.objects)>4: raise ValueError('At most four principal characters')
        ids={o.id for o in self.objects}
        for r in self.relations:
            if r.subject not in ids or r.target not in ids: raise ValueError('Relation references missing object')
        return self
class SceneRequest(StrictModel):
    script: str = Field(min_length=10,max_length=8000)

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
