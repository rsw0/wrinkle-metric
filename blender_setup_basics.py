import bpy
import bmesh
import numpy as np

#bpy.ops.object.mode_set(mode = 'OBJECT')  #enter objet mode

# delete all objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# create light
bpy.ops.object.light_add(type='SUN', radius=1, align='WORLD', location=(0, 0, 100), scale=(1, 1, 1))

# create background
bpy.ops.mesh.primitive_plane_add(size=16, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
background = bpy.context.active_object
background.name = "background"

background.modifiers.new(name = 'collision physics', type = 'COLLISION')
collision_physics = bpy.data.objects['background'].collision
collision_physics.thickness_outer = 0
collision_physics.cloth_friction = 2.0

# create material
background_mat = bpy.data.materials.new(name = "background material")
background.data.materials.append(background_mat)

bpy.ops.mesh.primitive_plane_add(size=2, enter_editmode=False, align='WORLD', location=(0, 0, 0.05), scale=(1, 1, 1))

cloth = bpy.context.active_object
cloth.name = "cloth"

# create material
bpy.ops.object.mode_set(mode = 'OBJECT')
cloth_mat = bpy.data.materials.new(name = "cloth_mat")
cloth.data.materials.append(cloth_mat)
bpy.context.object.active_material.diffuse_color = (0, 0.02, 0.8, 1)

num_cuts = 100
bpy.ops.object.mode_set(mode = 'EDIT')
bpy.ops.mesh.subdivide(number_cuts = num_cuts)
bpy.ops.object.mode_set(mode = 'OBJECT') #enter objet mode
bpy.ops.object.shade_smooth()
#bpy.context.object.color = (0.02, 0.02, 1, 1)

cloth_physics = cloth.modifiers.new(name = 'cloth physics', type = 'CLOTH')

cloth_physics.settings.tension_stiffness = 10
cloth_physics.settings.compression_stiffness = 10.0

cloth_physics.collision_settings.use_self_collision = True
cloth_physics.collision_settings.self_friction = 2
cloth_physics.collision_settings.distance_min = 0.001
cloth_physics.collision_settings.self_distance_min = 0.001


bpy.ops.object.modifier_add(type = 'SUBSURF')

bpy.ops.object.mode_set(mode = 'EDIT') #enter edit mode
bpy.ops.object.vertex_group_add()

# acquire vertices
x = 1 # 0 ~ num_cuts + 1
y = 1 # 0 ~ num_cuts + 1

def select_single(mesh, x, y, num_cuts):
    if x > num_cuts + 1 or y > num_cuts + 1:
        print("Max index exceeded.")
        vertex_index = 0
    else:
        if x == 0: 
            if y == 0:
                vertex_index = 0
            elif y == num_cuts + 1:
                vertex_index = 2
            else:
                vertex_index = num_cuts + 4 - y     
        elif x == num_cuts + 1: 
            if y == 0:
                vertex_index = 1
            elif y == num_cuts + 1:
                vertex_index = 3
            else:
                vertex_index = y + 2 * num_cuts + 3
        elif y == 0 and x > 0:
            vertex_index = x + num_cuts + 3    
        elif y == num_cuts + 1 and x > 0:
            vertex_index = 4 * num_cuts + 4 - x
        else:
            vertex_index = num_cuts * (x - 1) + 4 * num_cuts + 3 + y
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_mode(type = "VERT")
    bpy.ops.mesh.select_all(action = 'DESELECT')
    bpy.ops.object.mode_set(mode = 'OBJECT')
    mesh.data.vertices[vertex_index].select = True

select_single(cloth, x, y, num_cuts)
bpy.ops.object.mode_set(mode = 'EDIT') 
bpy.ops.object.vertex_group_assign()
cloth_physics.settings.vertex_group_mass = "Group"
cloth_physics.settings.pin_stiffness = 2

# create hook
bpy.ops.object.hook_add_newob()
bpy.ops.object.modifier_move_to_index(modifier="Hook-Empty", index=0)
hook = bpy.context.active_object

# animation
# dx_grid = 40 
# dy_grid = 40
dx = 1
dy = 1
dz = 0.5

bpy.ops.object.mode_set(mode = 'OBJECT') 

scn = bpy.context.scene
scn.frame_start = 1
scn.frame_end = 70

scn.frame_current = 0 
bpy.ops.anim.keyframe_insert_by_name(type="Location")

scn.frame_current = 5 # fisrt keyframe
bpy.ops.anim.keyframe_insert_by_name(type="Location")

scn.frame_current = 35 # second keyframe
bpy.ops.transform.translate(value = (dx/2, dy/2, dz))
bpy.ops.anim.keyframe_insert_by_name(type="Location")

scn.frame_current = 65 # thrid keyframe
bpy.ops.transform.translate(value = (dx/2, dy/2, -dz-0.05))
bpy.ops.anim.keyframe_insert_by_name(type="Location")

for i in range(71):
    scn.frame_set(i) # calculate stepwise  

depgraph = bpy.context.evaluated_depsgraph_get()

# define new bmesh object:
bm = bmesh.new()
bm.verts.ensure_lookup_table()
# read the evaluated (deformed) mesh data into the bmesh object:
bm.from_object(cloth , depgraph )
sum_of_z = 0
# iterate the bmesh verts:
for i, v in enumerate(bm.verts):
    v.co[2] += 0.05 # calibrate thickness
    sum_of_z += v.co[2]
    #print("frame: 50, vert: {}, location: {}".format(i, v.co))

print("The sum of z-value of all vertices is %.3f." %sum_of_z)

for area in bpy.context.screen.areas:
    if area.type == 'VIEW_3D':
        ctx = {
            "window": bpy.context.window, # current window, could also copy context
            "area": area, # our 3D View (the first found only actually)
            "region": None # just to suppress PyContext warning, doesn't seem to have any effect
        }
        bpy.ops.view3d.view_axis(ctx, type='TOP', align_active=True)
        area.spaces.active.region_3d.update() 

#add camera
bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 5), rotation=(0, -0, 0), scale=(1, 1, 1))
scn.camera = bpy.context.object

bpy.context.scene.render.image_settings.file_format='JPEG'
bpy.context.scene.render.filepath = "C:/Users/dongx/Desktop/render_result.jpg"
bpy.ops.render.render(write_still = True, use_viewport = True)