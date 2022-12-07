import bpy
import bmesh
import numpy as np

'''
Simulate randomized cloth folding processes in Blender 3.3
Output figures to a folder.
'''

# ----------- PARAMETERS -----------
NOC = 60 # NUMBER OF CUT
N = 1 # NUMBER OF RANDOM SAMPLES
X, Y = 0, 0 # PICK POSITION
dx, dy = 1.5, 1.5 # DISTANCE TO PLACE
ALPHA = 0.01 # threshold
# ----------------------------------

dz = np.random.randint(low = 10, high = 100, size = N) / 100 # randomize pick-up height
MID_FRAME = 5 + np.random.randint(low = 10, high = 60, size = N) 
END_FRAME = MID_FRAME * 2 - 5 # randomize folding speed
sumz_values = []

for k in range(N):
    def initialize(fabric_size = 2, desk_friction = 5.0, tension_stiffness = 30.0, compression_stiffness = 30.0, cloth_self_friction = 2.0):
        '''
        Create a desktop as the background and a square plane mesh as the cloth.
        The size of the cloth = fabric_size^2 (unit: m^2) 
        Subdivide the cloth into NOC^2 mesh grids and (NOC+1)^2 vertices
        Then assign physical properties to the desk and the cloth
        Return the cloth object and its physical properties
        '''
        bpy.ops.object.mode_set(mode = 'OBJECT')
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
        collision_physics.cloth_friction = desk_friction

        # create material
        background_mat = bpy.data.materials.new(name = "background material")
        background.data.materials.append(background_mat)

        bpy.ops.mesh.primitive_plane_add(size=fabric_size, enter_editmode=False, align='WORLD', location=(0, 0, 0.05), scale=(1, 1, 1))

        cloth = bpy.context.active_object
        cloth.name = "cloth"

        # create material
        bpy.ops.object.mode_set(mode = 'OBJECT')
        cloth_mat = bpy.data.materials.new(name = "cloth_mat")
        cloth.data.materials.append(cloth_mat)
        bpy.context.object.active_material.diffuse_color = (0, 0.02, 0.8, 1)

        bpy.ops.object.mode_set(mode = 'EDIT')
        bpy.ops.mesh.subdivide(number_cuts = NOC)
        bpy.ops.object.mode_set(mode = 'OBJECT') #enter objet mode
        bpy.ops.object.shade_smooth()
        #bpy.context.object.color = (0.02, 0.02, 1, 1)

        cloth_physics = cloth.modifiers.new(name = 'cloth physics', type = 'CLOTH')

        cloth_physics.settings.tension_stiffness = tension_stiffness
        cloth_physics.settings.compression_stiffness = compression_stiffness

        cloth_physics.collision_settings.use_self_collision = True
        cloth_physics.collision_settings.self_friction = cloth_self_friction
        cloth_physics.collision_settings.distance_min = 0.002
        cloth_physics.collision_settings.self_distance_min = 0.002


        bpy.ops.object.modifier_add(type = 'SUBSURF')
        
        return cloth, cloth_physics

    def select_single(mesh, physics, x, y, num_cuts = NOC):
        '''
        Select a single grid (coordinate x, y) on the flattened cloth 
        '''
        
        bpy.ops.object.mode_set(mode = 'EDIT') #enter edit mode
        bpy.ops.object.vertex_group_add()
        Unpin_group = mesh.vertex_groups.active
        Unpin_group.name = "Unpin"
        bpy.ops.mesh.select_mode(type = "VERT")
        bpy.ops.mesh.select_all(action = 'DESELECT')
        bpy.ops.object.vertex_group_assign()
        
        bpy.ops.object.vertex_group_add()
        mesh.vertex_groups.active.name = "Pin"
        Pin_group = mesh.vertex_groups.active
        
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
        bpy.ops.object.mode_set(mode = 'EDIT') 
        bpy.ops.object.vertex_group_assign()
        bpy.context.scene.tool_settings.vertex_group_weight = 0.99
        physics.settings.vertex_group_mass = "Pin"
        physics.settings.pin_stiffness = 1
        
        return Pin_group, Unpin_group
    
    
    def weight_group(mesh, pin_group):
        weight_mod = mesh.modifiers.new(name="Weight", type="VERTEX_WEIGHT_EDIT")
        weight_mod.vertex_group = pin_group.name
        weight_mod.remove_threshold = 1
        bpy.ops.object.modifier_move_to_index(modifier="Weight", index=0)
        return weight_mod


    def fold_cloth(dx, dy, dz, weight_mod, start_frame = 5, mid_frame = MID_FRAME[k], end_frame = END_FRAME[k]):
        '''
        Simulate the folding process by rendering an animation.
        The hook moves from the initial position (frame 5) to (dx/2, dy/2, dz) on mid_frame through a linear path,
        then moves to (dx, dy, 0) on end_frame through a linear path.
        The hook releases the cloth at the end.
        Return the scene object.
        '''
        
        # create hook
        bpy.ops.object.hook_add_newob()
        bpy.ops.object.modifier_move_to_index(modifier="Hook-Empty", index=0)
        hook_modifier = bpy.context.object.modifiers["Hook-Empty"]
        hook = bpy.context.active_object
        bpy.ops.object.mode_set(mode = 'OBJECT') 

        scn = bpy.context.scene
        scn.frame_start = 1
        scn.frame_end = end_frame + 30
        
        weight_mod.use_remove = False
        weight_mod.keyframe_insert("use_remove", frame = 1)

        weight_mod.use_remove = True
        weight_mod.keyframe_insert("use_remove", frame = end_frame + 5)
        
        scn.frame_current = 0 
        bpy.ops.anim.keyframe_insert_by_name(type="Location")

        scn.frame_current = start_frame # fisrt keyframe
        bpy.ops.anim.keyframe_insert_by_name(type="Location")

        scn.frame_current = mid_frame # second keyframe
        bpy.ops.transform.translate(value = (dx/2, dy/2, dz))
        bpy.ops.anim.keyframe_insert_by_name(type="Location")

        scn.frame_current = end_frame # thrid keyframe
        bpy.ops.transform.translate(value = (dx/2, dy/2, -dz))
        bpy.ops.anim.keyframe_insert_by_name(type="Location")

        for j in range(scn.frame_end + 1):
            scn.frame_set(j) # calculate stepwise 
            
        return scn

    cloth, cloth_physics = initialize()
    x = X # 0 ~ num_cuts + 1
    y = Y # 0 ~ num_cuts + 1
    pin, unpin = select_single(cloth, cloth_physics, x, y)
    weight_modifier = weight_group(cloth, pin)
    scn = fold_cloth(dx, dy, dz[k], weight_modifier)

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
        sum_of_z += (v.co[2] - ALPHA)
        #print("frame: 50, vert: {}, location: {}".format(i, v.co))

    print("The sum of z-value of all vertices is %.3f." %sum_of_z)
    sumz_values.append(sum_of_z)

    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            ctx = {
                "window": bpy.context.window, # current window, could also copy context
                "area": area, # our 3D View (the first found only actually)
                "region": None # just to suppress PyContext warning, doesn't seem to have any effect
            }
            bpy.ops.view3d.view_axis(ctx, type='TOP', align_active=True)
            area.spaces.active.region_3d.update() 

    # add camera
    bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(0, 0, 6), rotation=(0, -0, 0), scale=(1, 1, 1))
    scn.camera = bpy.context.object

    # render and output images as jpeg files
    bpy.context.scene.render.image_settings.file_format='JPEG'
    bpy.context.scene.render.filepath = "/Users/donge/Documents/CU/6998 Robot Learning/blender/folded_fig/%d_dz_%.2f_frame_%d.jpg"%(k, dz[k], END_FRAME[k]) # YOUR OWN PATH
    bpy.ops.render.render(write_still = True, use_viewport = True)
